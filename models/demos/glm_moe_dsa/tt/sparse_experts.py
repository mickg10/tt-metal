# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sparse expert computation using DRAMStreamingExpertsMatmul."""

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.glm_moe_dsa.b1_utils import (
    get_pinned_optimal_dram_bank_to_logical_worker_assignment,
    pad_n_to_dram_banks,
    shuffle_dram_tiles,
)
from models.demos.glm_moe_dsa.micro_ops.dram_streaming_experts_matmul.op import (
    DRAMStreamingExpertsMatmul,
)

TILE_SIZE = 32
BH_NUM_DRAM_BANKS = 8


def convert_expert_weights_sparse(
    expert_weights_torch, num_experts_per_device, mesh_device, dtype, cache_path=None, tag="w",
):
    """Convert expert weights to WIDTH_SHARDED format for DRAMStreamingExpertsMatmul."""
    num_devices = mesh_device.get_num_devices()
    total_experts = expert_weights_torch.shape[1]
    assert total_experts == num_experts_per_device * num_devices

    _, _, K, N = expert_weights_torch.shape
    num_banks = BH_NUM_DRAM_BANKS
    N_padded = pad_n_to_dram_banks(N, TILE_SIZE, num_banks)
    per_core_N = N_padded // num_banks

    logger.info(f"Sparse [{tag}]: K={K}, N={N}, N_padded={N_padded}, dtype={dtype}, E/dev={num_experts_per_device}")

    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    expert_tensors = []
    for e_idx in range(num_experts_per_device):
        per_dev = []
        for d_id in range(num_devices):
            g_id = d_id * num_experts_per_device + e_idx
            w = expert_weights_torch[0, g_id].clone()
            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))
            w = shuffle_dram_tiles(w.unsqueeze(0), TILE_SIZE, num_banks).reshape(1, 1, K, N_padded)
            per_dev.append(w)
        stacked = torch.cat(per_dev, dim=0)
        cf = str(cache_path / f"{tag}_e{e_idx}_ws") if cache_path else None
        et = ttnn.as_tensor(stacked, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                            memory_config=mem_config, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
                            cache_file_name=cf)
        expert_tensors.append(et)
        if (e_idx + 1) % 4 == 0:
            logger.info(f"  [{tag}] {e_idx + 1}/{num_experts_per_device} experts uploaded")
    logger.info(f"  [{tag}] All {num_experts_per_device} experts uploaded (WIDTH_SHARDED)")
    return expert_tensors


def sparse_expert_forward(
    x, gate_expert_tensors, up_expert_tensors, down_expert_tensors,
    num_experts_per_device, hidden_size, intermediate_size, device,
    selected_experts_k=8, expert_indices=None,
):
    """Sparse expert forward using DRAMStreamingExpertsMatmul.

    Processes dispatch output through gate(SiLU) + up(mul) + down pipeline.
    Uses eager mode with per-token tiny-tile processing.

    x: [1, 1, total_tokens, hidden_size] in TILE layout
    Returns: [1, num_experts_per_device, total_tokens, hidden_size]
    """
    _, _, total_tokens, _ = x.shape
    num_banks = BH_NUM_DRAM_BANKS
    N_gate = pad_n_to_dram_banks(intermediate_size, TILE_SIZE, num_banks)
    N_down = pad_n_to_dram_banks(hidden_size, TILE_SIZE, num_banks)
    per_core_N_gate = N_gate // num_banks
    per_core_N_down = N_down // num_banks

    all_workers = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    num_cores = len(all_workers)
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_workers])

    # subblock_k for gate/up (K=hidden_size)
    Kt_gate = hidden_size // TILE_SIZE
    sk_gate = Kt_gate // 2
    while Kt_gate % sk_gate != 0 or sk_gate % 2 != 0:
        sk_gate -= 1

    # subblock_k for down (K=intermediate_size)
    Kt_down = intermediate_size // TILE_SIZE
    sk_down = Kt_down // 2
    while Kt_down % sk_down != 0 or sk_down % 2 != 0:
        sk_down -= 1

    in0_tile = ttnn.Tile([1, 32])
    out_tile = ttnn.Tile([1, 32])
    idx_tile = ttnn.Tile([1, 16])
    in1_tile = ttnn.Tile([32, 32])

    # Index tensor: [0,1,...,K-1] for all experts
    idx_data = torch.zeros(num_cores, 16, dtype=torch.int32)
    for e in range(min(selected_experts_k, 16)):
        idx_data[:, e] = e
    idx_shard = ttnn.ShardSpec(cg, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    idx_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, idx_shard)
    idx_t = ttnn.from_torch(idx_data.to(torch.uint16), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=idx_mem, tile=idx_tile,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(device))

    # Working buffer for gate/up
    cb_tiles_gate = sk_gate * 3
    wb_w_gate = cb_tiles_gate * 32 * num_cores
    wb_spec_gate = ttnn.ShardSpec(cg, (32, cb_tiles_gate * 32), ttnn.ShardOrientation.ROW_MAJOR)
    wb_mem_gate = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, wb_spec_gate)
    gate_dtype = gate_expert_tensors[0].dtype
    wb_gate = ttnn.from_torch(torch.zeros(1, 1, 32, wb_w_gate).bfloat16(), dtype=gate_dtype,
                               layout=ttnn.TILE_LAYOUT, device=device, memory_config=wb_mem_gate, tile=in1_tile,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(device))

    # Working buffer for down
    cb_tiles_down = sk_down * 3
    wb_w_down = cb_tiles_down * 32 * num_cores
    wb_spec_down = ttnn.ShardSpec(cg, (32, cb_tiles_down * 32), ttnn.ShardOrientation.ROW_MAJOR)
    wb_mem_down = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, wb_spec_down)
    down_dtype = down_expert_tensors[0].dtype
    wb_down = ttnn.from_torch(torch.zeros(1, 1, 32, wb_w_down).bfloat16(), dtype=down_dtype,
                               layout=ttnn.TILE_LAYOUT, device=device, memory_config=wb_mem_down, tile=in1_tile,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(device))

    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    token_outputs = []

    for t in range(total_tokens):
        # Extract token: [1, 1, 1, hidden]
        tok = ttnn.slice(x_rm, [0, 0, t, 0], [1, 1, t + 1, hidden_size])
        tok_rep = ttnn.repeat(tok, ttnn.Shape((1, 1, num_cores, 1)))
        ttnn.deallocate(tok)
        in0_spec = ttnn.ShardSpec(cg, [1, hidden_size], ttnn.ShardOrientation.ROW_MAJOR)
        in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_spec)
        tok_hs = ttnn.to_memory_config(ttnn.to_layout(tok_rep, ttnn.TILE_LAYOUT, tile=in0_tile), in0_mem)
        ttnn.deallocate(tok_rep)

        # Gate output
        gate_out_w = per_core_N_gate * selected_experts_k
        gate_out_spec = ttnn.ShardSpec(cg, [1, gate_out_w], ttnn.ShardOrientation.ROW_MAJOR)
        gate_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_out_spec)
        gate_n_total = N_gate * selected_experts_k
        gate_out = ttnn.from_torch(torch.zeros(1, 1, 1, gate_n_total).bfloat16(), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_out_mem, tile=out_tile,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(device))

        gate_out = DRAMStreamingExpertsMatmul.op(
            input_a=tok_hs, input_b=gate_expert_tensors[0], output_tensor=gate_out,
            fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.LoFi,
            fused_activation="silu", index_tensor=idx_t,
            selected_experts_k=selected_experts_k, subblock_k=sk_gate,
            working_buf_tensor=wb_gate,
        )

        # Up + mul with gate
        mm_out = ttnn.from_torch(torch.zeros(1, 1, 1, gate_n_total).bfloat16(), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_out_mem, tile=out_tile,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(device))
        activated = ttnn.from_torch(torch.zeros(1, 1, 1, gate_n_total).bfloat16(), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_out_mem, tile=out_tile,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(device))

        activated = DRAMStreamingExpertsMatmul.op(
            input_a=tok_hs, input_b=up_expert_tensors[0], output_tensor=activated,
            fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.LoFi,
            index_tensor=idx_t, selected_experts_k=selected_experts_k,
            subblock_k=sk_gate, mul_tensor=gate_out, mm_out_tensor=mm_out,
            working_buf_tensor=wb_gate,
        )
        ttnn.deallocate(gate_out)
        ttnn.deallocate(mm_out)
        ttnn.deallocate(tok_hs)

        # Down proj per expert
        act_i = ttnn.to_memory_config(activated, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(activated)
        act_i = ttnn.reshape(act_i, [selected_experts_k, 1, 1, intermediate_size])

        down_outputs = []
        for e in range(selected_experts_k):
            e_act = ttnn.slice(act_i, [e, 0, 0, 0], [e + 1, 1, 1, intermediate_size])
            e_act = ttnn.reshape(e_act, [1, 1, 1, intermediate_size])
            e_act_rep = ttnn.repeat(e_act, ttnn.Shape((1, 1, num_cores, 1)))
            ttnn.deallocate(e_act)
            in0_down_spec = ttnn.ShardSpec(cg, [1, intermediate_size], ttnn.ShardOrientation.ROW_MAJOR)
            in0_down_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_down_spec)
            e_act_hs = ttnn.to_memory_config(ttnn.to_layout(e_act_rep, ttnn.TILE_LAYOUT, tile=in0_tile), in0_down_mem)
            ttnn.deallocate(e_act_rep)

            e_idx_data = torch.zeros(num_cores, 16, dtype=torch.int32)
            e_idx_data[:, 0] = e
            e_idx = ttnn.from_torch(e_idx_data.to(torch.uint16), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=idx_mem, tile=idx_tile,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(device))

            d_out_spec = ttnn.ShardSpec(cg, [1, per_core_N_down], ttnn.ShardOrientation.ROW_MAJOR)
            d_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, d_out_spec)
            d_out = ttnn.from_torch(torch.zeros(1, 1, 1, N_down).bfloat16(), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device, memory_config=d_out_mem, tile=out_tile,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(device))

            d_out = DRAMStreamingExpertsMatmul.op(
                input_a=e_act_hs, input_b=down_expert_tensors[0], output_tensor=d_out,
                fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.HiFi2,
                index_tensor=e_idx, selected_experts_k=1, subblock_k=sk_down,
                working_buf_tensor=wb_down,
            )
            ttnn.deallocate(e_act_hs)
            ttnn.deallocate(e_idx)
            d_out_i = ttnn.to_memory_config(d_out, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(d_out)
            down_outputs.append(d_out_i)

        ttnn.deallocate(act_i)

        if len(down_outputs) == 1:
            tok_out = down_outputs[0]
        else:
            tok_out = ttnn.concat(down_outputs, dim=0)
            for d in down_outputs:
                ttnn.deallocate(d)

        if selected_experts_k < num_experts_per_device:
            pad = ttnn.from_torch(
                torch.zeros(num_experts_per_device - selected_experts_k, 1, 1, hidden_size, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device))
            tok_out = ttnn.concat([tok_out, pad], dim=0)
            ttnn.deallocate(pad)

        token_outputs.append(tok_out)

    ttnn.deallocate(x_rm)
    ttnn.deallocate(idx_t)
    ttnn.deallocate(wb_gate)
    ttnn.deallocate(wb_down)

    if len(token_outputs) == 1:
        result = token_outputs[0]
    else:
        result = ttnn.concat(token_outputs, dim=2)
        for t in token_outputs:
            ttnn.deallocate(t)

    result = ttnn.permute(result, (1, 0, 2, 3))
    return result

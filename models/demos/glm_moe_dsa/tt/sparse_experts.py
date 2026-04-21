# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sparse expert computation using DRAMStreamingExpertsMatmul.

This kernel is a decode-row kernel designed for M=1 (single token).
It requires Tile([1, 32]) for input and output (NOT standard 32x32 tiles).
custom_mm_block only supports in0 tile shapes {1,2,4,8}x32.

At bs=1 decode with EP=32 and 8 local experts, only ~1 expert fires per device.
Dense path wastes 87.5% compute. Sparse path computes only the needed expert(s).
"""

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
    expert_weights_torch: torch.Tensor,
    num_experts_per_device: int,
    mesh_device,
    dtype: ttnn.DataType,
    cache_path: Path = None,
    tag: str = "w",
):
    """Convert expert weights to WIDTH_SHARDED format for DRAMStreamingExpertsMatmul.

    Each expert uploaded as separate WIDTH_SHARDED tensor. Sequential allocation
    ensures contiguous DRAM placement for indexed access.
    """
    num_devices = mesh_device.get_num_devices()
    total_experts = expert_weights_torch.shape[1]
    assert total_experts == num_experts_per_device * num_devices

    _, _, K, N = expert_weights_torch.shape
    num_banks = BH_NUM_DRAM_BANKS
    N_padded = pad_n_to_dram_banks(N, TILE_SIZE, num_banks)
    per_core_N = N_padded // num_banks

    logger.info(
        f"Sparse expert weights [{tag}]: K={K}, N={N}, N_padded={N_padded}, "
        f"per_core_N={per_core_N}, dtype={dtype}, experts_per_device={num_experts_per_device}"
    )

    dram_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))}
    )
    shard_spec = ttnn.ShardSpec(dram_shard_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec
    )

    expert_tensors = []
    for expert_local_idx in range(num_experts_per_device):
        per_device_weights = []
        for device_id in range(num_devices):
            global_expert_id = device_id * num_experts_per_device + expert_local_idx
            w = expert_weights_torch[0, global_expert_id].clone()
            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))
            w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), TILE_SIZE, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)
            per_device_weights.append(w_shuffled)

        stacked = torch.cat(per_device_weights, dim=0)
        cache_file = str(cache_path / f"{tag}_expert_{expert_local_idx}_sparse_ws") if cache_path else None

        expert_t = ttnn.as_tensor(
            stacked, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=mem_config, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            cache_file_name=cache_file,
        )
        expert_tensors.append(expert_t)
        if (expert_local_idx + 1) % 4 == 0:
            logger.info(f"  [{tag}] Uploaded {expert_local_idx + 1}/{num_experts_per_device} experts")

    logger.info(f"  [{tag}] All {num_experts_per_device} experts uploaded (WIDTH_SHARDED)")
    return expert_tensors


def _get_compute_core_grid(device):
    """Get pinned compute core grid for BH Galaxy mesh compatibility."""
    all_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    num_cores = len(all_worker_cores)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
    )
    return all_worker_cores, num_cores, compute_core_grid


def sparse_expert_forward(
    x: ttnn.Tensor,
    gate_expert_tensors: list,
    up_expert_tensors: list,
    down_expert_tensors: list,
    num_experts_per_device: int,
    hidden_size: int,
    intermediate_size: int,
    device,
    selected_experts_k: int = 8,
    expert_indices: ttnn.Tensor = None,
):
    """Forward pass using DRAMStreamingExpertsMatmul for sparse expert computation.

    Input x: [1, 1, total_tokens, hidden_size] in TILE layout (standard 32x32 tiles)
    Output: [1, num_experts_per_device, total_tokens, hidden_size] in TILE layout

    For each dispatched token, computes the expert MLP:
      gate = SiLU(token @ gate_weights[expert_idx])
      up = token @ up_weights[expert_idx]
      activated = gate * up
      down = activated @ down_weights[expert_idx]

    With selected_experts_k=8, ALL experts are computed (no routing needed).
    With selected_experts_k=1, only the selected expert is computed (8x savings).
    """
    _, _, total_tokens, _ = x.shape
    num_banks = BH_NUM_DRAM_BANKS
    N_gate = pad_n_to_dram_banks(intermediate_size, TILE_SIZE, num_banks)
    N_down = pad_n_to_dram_banks(hidden_size, TILE_SIZE, num_banks)
    per_core_N_gate = N_gate // num_banks
    per_core_N_down = N_down // num_banks

    all_worker_cores, num_cores, compute_core_grid = _get_compute_core_grid(device)

    # Convert input from standard TILE to ROW_MAJOR for token extraction
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    # Process each token through the expert MLP
    # For bs=1 decode, total_tokens is small (e.g., 8 for dispatch across 8 devices)
    token_outputs = []  # Each: [num_experts_per_device, 1, 1, hidden_size]

    for t_idx in range(total_tokens):
        # Extract single token: [1, 1, 1, hidden_size]
        token = ttnn.slice(x_rm, [0, 0, t_idx, 0], [1, 1, t_idx + 1, hidden_size])

        # Convert to HEIGHT_SHARDED with tiny tile [1, 32], replicated on compute cores
        token_replicated = ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT)
        # Repeat for each core: [1, 1, num_cores, hidden_size]
        token_rep = ttnn.repeat(token_replicated, ttnn.Shape((1, 1, num_cores, 1)))
        ttnn.deallocate(token_replicated)

        in0_tile = ttnn.Tile([1, 32])
        in0_spec = ttnn.ShardSpec(compute_core_grid, [1, hidden_size], ttnn.ShardOrientation.ROW_MAJOR)
        in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_spec)
        token_hs = ttnn.to_memory_config(
            ttnn.to_layout(token_rep, ttnn.TILE_LAYOUT, tile=in0_tile), in0_mem
        )
        ttnn.deallocate(token_rep)
        ttnn.deallocate(token)

        # Index tensor: sequential [0,1,...,K-1] for all-experts mode
        if expert_indices is None:
            idx_data = torch.zeros(num_cores, 16, dtype=torch.int32)
            for e in range(min(selected_experts_k, 16)):
                idx_data[:, e] = e
            idx_data = idx_data.to(torch.uint16)
            idx_tile = ttnn.Tile([1, 16])
            idx_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
            idx_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, idx_spec)
            idx_t = ttnn.from_torch(
                idx_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=idx_mem, tile=idx_tile,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            idx_t = expert_indices

        # subblock_k calculation
        K_gate = hidden_size
        Kt_gate = K_gate // TILE_SIZE
        subblock_k_gate = Kt_gate // 2
        while Kt_gate % subblock_k_gate != 0 or subblock_k_gate % 2 != 0:
            subblock_k_gate -= 1

        # Working buffer for CB1
        in1_tile_obj = ttnn.Tile([32, 32])
        num_in1_buffers = 3
        in1_CB_tiles = subblock_k_gate * num_in1_buffers
        wb_shard = (32, in1_CB_tiles * 32)
        wb_total_w = in1_CB_tiles * 32 * num_cores
        wb_spec = ttnn.ShardSpec(compute_core_grid, wb_shard, ttnn.ShardOrientation.ROW_MAJOR)
        wb_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, wb_spec)
        gate_dtype = gate_expert_tensors[0].dtype
        wb_gate = ttnn.from_torch(
            torch.zeros([1, 1, 32, wb_total_w]).bfloat16(), dtype=gate_dtype,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=wb_mem, tile=in1_tile_obj,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        # ===== Gate proj: SiLU fused =====
        gate_out_width = per_core_N_gate * selected_experts_k
        gate_out_spec = ttnn.ShardSpec(compute_core_grid, [1, gate_out_width], ttnn.ShardOrientation.ROW_MAJOR)
        gate_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_out_spec)
        gate_total_n = N_gate * selected_experts_k
        out_tile = ttnn.Tile([1, 32])
        gate_out = ttnn.from_torch(
            torch.zeros([1, 1, 1, gate_total_n]).bfloat16(), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_out_mem, tile=out_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        gate_out = DRAMStreamingExpertsMatmul.op(
            input_a=token_hs, input_b=gate_expert_tensors[0], output_tensor=gate_out,
            fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.LoFi,
            fused_activation="silu", index_tensor=idx_t,
            selected_experts_k=selected_experts_k, subblock_k=subblock_k_gate,
            working_buf_tensor=wb_gate,
        )

        # ===== Up proj: multiply with gate output =====
        mm_out = ttnn.from_torch(
            torch.zeros([1, 1, 1, gate_total_n]).bfloat16(), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_out_mem, tile=out_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        activated = ttnn.from_torch(
            torch.zeros([1, 1, 1, gate_total_n]).bfloat16(), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_out_mem, tile=out_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        up_dtype = up_expert_tensors[0].dtype
        # Reuse same working buffer if same dtype, else create new
        if up_dtype == gate_dtype:
            wb_up = wb_gate
        else:
            wb_up = ttnn.from_torch(
                torch.zeros([1, 1, 32, wb_total_w]).bfloat16(), dtype=up_dtype,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=wb_mem, tile=in1_tile_obj,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

        activated = DRAMStreamingExpertsMatmul.op(
            input_a=token_hs, input_b=up_expert_tensors[0], output_tensor=activated,
            fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.LoFi,
            index_tensor=idx_t, selected_experts_k=selected_experts_k,
            subblock_k=subblock_k_gate,
            mul_tensor=gate_out, mm_out_tensor=mm_out,
            working_buf_tensor=wb_up,
        )

        ttnn.deallocate(gate_out)
        ttnn.deallocate(mm_out)
        ttnn.deallocate(token_hs)
        ttnn.deallocate(wb_gate)
        if up_dtype != gate_dtype:
            ttnn.deallocate(wb_up)

        # ===== Down proj: per-expert =====
        # activated: [1, 1, 1, intermediate_size * selected_experts_k] WIDTH_SHARDED
        # Need to compute down_proj for each expert separately since input differs

        # Convert activated to interleaved for slicing
        activated_i = ttnn.to_memory_config(activated, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(activated)

        # Reshape to separate experts: [selected_experts_k, 1, 1, intermediate_size]
        activated_i = ttnn.reshape(activated_i, [selected_experts_k, 1, 1, intermediate_size])

        down_outputs = []
        for e in range(selected_experts_k):
            # Slice expert e's activated output
            e_act = ttnn.slice(activated_i, [e, 0, 0, 0], [e + 1, 1, 1, intermediate_size])
            e_act = ttnn.reshape(e_act, [1, 1, 1, intermediate_size])

            # Convert to HEIGHT_SHARDED tiny tile for down_proj input
            e_act_rep = ttnn.repeat(e_act, ttnn.Shape((1, 1, num_cores, 1)))
            ttnn.deallocate(e_act)
            K_down = intermediate_size
            in0_down_spec = ttnn.ShardSpec(compute_core_grid, [1, K_down], ttnn.ShardOrientation.ROW_MAJOR)
            in0_down_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_down_spec)
            e_act_hs = ttnn.to_memory_config(
                ttnn.to_layout(e_act_rep, ttnn.TILE_LAYOUT, tile=ttnn.Tile([1, 32])), in0_down_mem
            )
            ttnn.deallocate(e_act_rep)

            # Per-expert index
            e_idx_data = torch.zeros(num_cores, 16, dtype=torch.int32)
            e_idx_data[:, 0] = e
            e_idx_data = e_idx_data.to(torch.uint16)
            e_idx = ttnn.from_torch(
                e_idx_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=idx_mem, tile=idx_tile,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

            # Down proj output
            down_out_spec = ttnn.ShardSpec(compute_core_grid, [1, per_core_N_down], ttnn.ShardOrientation.ROW_MAJOR)
            down_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, down_out_spec)
            down_out = ttnn.from_torch(
                torch.zeros([1, 1, 1, N_down]).bfloat16(), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=down_out_mem, tile=out_tile,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

            # Down proj subblock_k
            Kt_down = K_down // TILE_SIZE
            subblock_k_down = Kt_down // 2
            while Kt_down % subblock_k_down != 0 or subblock_k_down % 2 != 0:
                subblock_k_down -= 1

            # Working buffer for down
            down_dtype = down_expert_tensors[0].dtype
            in1_CB_tiles_down = subblock_k_down * 3
            wb_down_w = in1_CB_tiles_down * 32 * num_cores
            wb_down_spec = ttnn.ShardSpec(compute_core_grid, (32, in1_CB_tiles_down * 32), ttnn.ShardOrientation.ROW_MAJOR)
            wb_down_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, wb_down_spec)
            wb_down = ttnn.from_torch(
                torch.zeros([1, 1, 32, wb_down_w]).bfloat16(), dtype=down_dtype,
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=wb_down_mem, tile=in1_tile_obj,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

            down_out = DRAMStreamingExpertsMatmul.op(
                input_a=e_act_hs, input_b=down_expert_tensors[0], output_tensor=down_out,
                fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.HiFi2,
                index_tensor=e_idx, selected_experts_k=1,
                subblock_k=subblock_k_down, working_buf_tensor=wb_down,
            )

            ttnn.deallocate(e_act_hs)
            ttnn.deallocate(e_idx)
            ttnn.deallocate(wb_down)

            # Convert to interleaved
            down_out_i = ttnn.to_memory_config(down_out, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(down_out)
            down_outputs.append(down_out_i)

        ttnn.deallocate(activated_i)
        if expert_indices is None:
            ttnn.deallocate(idx_t)

        # Stack expert outputs: [selected_experts_k, 1, 1, hidden_size]
        if len(down_outputs) == 1:
            token_expert_out = down_outputs[0]
        else:
            token_expert_out = ttnn.concat(down_outputs, dim=0)
            for t in down_outputs:
                ttnn.deallocate(t)

        # Pad to num_experts_per_device if needed
        if selected_experts_k < num_experts_per_device:
            pad = ttnn.from_torch(
                torch.zeros([num_experts_per_device - selected_experts_k, 1, 1, hidden_size], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            token_expert_out = ttnn.concat([token_expert_out, pad], dim=0)
            ttnn.deallocate(pad)

        token_outputs.append(token_expert_out)

    ttnn.deallocate(x_rm)

    # Stack all token outputs: each is [num_experts_per_device, 1, 1, hidden_size]
    # Need final shape: [1, num_experts_per_device, total_tokens, hidden_size]
    if len(token_outputs) == 1:
        result = token_outputs[0]  # [num_experts_per_device, 1, 1, hidden_size]
    else:
        result = ttnn.concat(token_outputs, dim=2)  # [num_experts_per_device, 1, total_tokens, hidden_size]
        for t in token_outputs:
            ttnn.deallocate(t)

    # Permute to [1, num_experts_per_device, total_tokens, hidden_size]
    result = ttnn.permute(result, (1, 0, 2, 3))

    return result

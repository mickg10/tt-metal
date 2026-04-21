# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sparse expert computation using DRAMStreamingExpertsMatmul.

Replaces the dense expert path (repeat tokens × all experts → batched matmul)
with DRAM-streaming matmul that reads only the selected expert's weights from DRAM.

At bs=1 decode with EP=32 and 8 local experts, only ~1 expert fires per device.
Dense path wastes 87.5% compute. Sparse path computes only the needed expert(s).

Weight format: each expert stored as individual WIDTH_SHARDED tensor in DRAM with
column-major tile order per bank shard. Experts allocated contiguously so the kernel
can use base_addr + expert_idx * expert_size_bytes for indexed access.
"""

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.glm_moe_dsa.b1_utils import (
    pad_n_to_dram_banks,
    shuffle_dram_tiles,
)
from models.demos.glm_moe_dsa.micro_ops.dram_streaming_experts_matmul.op import (
    DRAMStreamingExpertsMatmul,
)

# Tile size for BH (standard 32x32 tiles)
TILE_SIZE = 32

# Number of DRAM banks on Blackhole
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

    Each expert's weights are uploaded as a separate WIDTH_SHARDED tensor in DRAM.
    Sequential allocation ensures contiguous placement — the kernel uses
    base_addr + expert_idx * expert_size_bytes to access different experts.

    Tile layout within each bank shard is column-major (K tiles contiguous per N column).

    Args:
        expert_weights_torch: [1, num_experts_total, K, N] tensor (already transposed)
        num_experts_per_device: Number of experts per device
        mesh_device: TTNN mesh device
        dtype: Weight data type (e.g., bfloat4_b)
        cache_path: Optional cache directory for weight files
        tag: Identifier for cache file naming

    Returns:
        List of ttnn tensors (one per local expert). First tensor is used as input_b
        for DRAMStreamingExpertsMatmul (base address for indexed access).
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

    # DRAM shard grid: bank coordinates (0,0) to (num_banks-1, 0)
    dram_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))}
    )
    shard_spec = ttnn.ShardSpec(
        dram_shard_grid,
        [K, per_core_N],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )

    # Upload each expert sequentially to ensure contiguous DRAM allocation.
    # CRITICAL: no other DRAM allocations should happen between expert uploads
    # on the same device, or the contiguity assumption breaks.
    expert_tensors = []

    for expert_local_idx in range(num_experts_per_device):
        # Build per-device weights for this local expert
        per_device_weights = []
        for device_id in range(num_devices):
            global_expert_id = device_id * num_experts_per_device + expert_local_idx
            w = expert_weights_torch[0, global_expert_id].clone()  # [K, N]

            # Pad N if needed
            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))

            # Shuffle tiles to column-major per bank shard
            w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), TILE_SIZE, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)
            per_device_weights.append(w_shuffled)

        # Stack for mesh mapper: [num_devices, 1, K, N_padded]
        stacked = torch.cat(per_device_weights, dim=0)

        cache_file = None
        if cache_path is not None:
            cache_file = str(cache_path / f"{tag}_expert_{expert_local_idx}_sparse_ws")

        expert_t = ttnn.as_tensor(
            stacked,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            cache_file_name=cache_file,
        )
        expert_tensors.append(expert_t)

        if (expert_local_idx + 1) % 4 == 0:
            logger.info(f"  [{tag}] Uploaded {expert_local_idx + 1}/{num_experts_per_device} experts")

    logger.info(f"  [{tag}] All {num_experts_per_device} experts uploaded to DRAM (WIDTH_SHARDED)")
    return expert_tensors


def create_sparse_expert_tensors(
    device,
    num_experts_per_device: int,
    M: int,
    K: int,
    N: int,
    in0_dtype=ttnn.bfloat16,
):
    """Pre-allocate input, output, and index tensors for DRAMStreamingExpertsMatmul.

    Args:
        device: TTNN device (single device or mesh)
        num_experts_per_device: Number of local experts
        M: Number of token rows (typically 32 for padded decode)
        K: Input dimension (hidden_size for gate/up, intermediate for down)
        N: Output dimension (intermediate for gate/up, hidden_size for down)
        in0_dtype: Data type for input tensor

    Returns:
        Dict with pre-allocated tensors and configs
    """
    num_banks = BH_NUM_DRAM_BANKS
    N_padded = pad_n_to_dram_banks(N, TILE_SIZE, num_banks)
    per_core_N = N_padded // num_banks

    # Compute cores from optimal DRAM bank assignment
    all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(all_worker_cores)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
    )

    # Input A: HEIGHT_SHARDED in L1, replicated on compute cores
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # Output: WIDTH_SHARDED in L1 on compute cores
    output_shard_spec = ttnn.ShardSpec(
        compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )

    # Index tensor: HEIGHT_SHARDED in L1, [1, 16] per core
    index_tile = ttnn.Tile([1, 16])
    index_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, index_shard_spec
    )

    return {
        "num_cores": num_cores,
        "compute_core_grid": compute_core_grid,
        "in0_mem_config": in0_mem_config,
        "output_mem_config": output_mem_config,
        "index_mem_config": index_mem_config,
        "index_tile": index_tile,
        "M": M,
        "K": K,
        "N": N,
        "N_padded": N_padded,
        "per_core_N": per_core_N,
        "all_worker_cores": all_worker_cores,
    }


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

    Args:
        x: Input tensor [1, 1, tokens, hidden_size] in TILE layout
        gate_expert_tensors: List of WIDTH_SHARDED gate_proj weight tensors (one per expert)
        up_expert_tensors: List of WIDTH_SHARDED up_proj weight tensors
        down_expert_tensors: List of WIDTH_SHARDED down_proj weight tensors
        num_experts_per_device: Number of local experts
        hidden_size: Model hidden size
        intermediate_size: Expert intermediate size
        device: TTNN device
        selected_experts_k: Number of experts to compute (1 for sparse, 8 for dense-via-streaming)
        expert_indices: Optional index tensor for sparse mode [num_cores, 16] uint16

    Returns:
        Expert output [1, num_experts_per_device, tokens, hidden_size] in TILE layout
    """
    num_banks = BH_NUM_DRAM_BANKS
    _, _, num_tokens, _ = x.shape

    # Padded dimensions
    N_gate = pad_n_to_dram_banks(intermediate_size, TILE_SIZE, num_banks)
    N_down = pad_n_to_dram_banks(hidden_size, TILE_SIZE, num_banks)
    per_core_N_gate = N_gate // num_banks
    per_core_N_down = N_down // num_banks

    # Compute cores
    all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(all_worker_cores)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
    )

    M = num_tokens  # Padded to tile boundary

    # ===== Prepare Input A: HEIGHT_SHARDED replicated on compute cores =====
    # x is [1, 1, M, K] in TILE layout. Need to replicate on each compute core.
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, [M, hidden_size], ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # Convert x to HEIGHT_SHARDED (replicated on compute cores)
    x_replicated = ttnn.to_memory_config(x, in0_mem_config)

    # ===== Prepare Index Tensor =====
    if expert_indices is None:
        # Default: sequential indices [0, 1, 2, ..., num_experts-1] for all-expert mode
        index_data = torch.zeros(num_cores, 16, dtype=torch.int32)
        for e in range(min(selected_experts_k, 16)):
            index_data[:, e] = e
        index_data = index_data.to(torch.uint16)

        index_tile = ttnn.Tile([1, 16])
        index_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
        index_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, index_shard_spec
        )
        expert_indices = ttnn.from_torch(
            index_data,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=index_mem_config,
            tile=index_tile,
        )

    # ===== Output tensor for gate_proj =====
    gate_output_width = per_core_N_gate * selected_experts_k
    gate_out_shard_spec = ttnn.ShardSpec(
        compute_core_grid, [M, gate_output_width], ttnn.ShardOrientation.ROW_MAJOR
    )
    gate_out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_out_shard_spec
    )
    gate_output_total_n = N_gate * selected_experts_k
    gate_output = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_output_total_n], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_out_mem_config,
    )

    # ===== Output tensor for up_proj =====
    up_output = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_output_total_n], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_out_mem_config,
    )

    # ===== Gate proj: SiLU fused =====
    K_tiles = hidden_size // TILE_SIZE
    subblock_k = K_tiles if K_tiles <= 8 else K_tiles // 2

    gate_output = DRAMStreamingExpertsMatmul.op(
        input_a=x_replicated,
        input_b=gate_expert_tensors[0],  # First expert tensor (base addr)
        output_tensor=gate_output,
        fp32_dest_acc_en=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fused_activation="silu",
        index_tensor=expert_indices,
        selected_experts_k=selected_experts_k,
        subblock_k=subblock_k,
    )

    # ===== Up proj: multiply with gate output =====
    # Use mul_tensor to fuse: output = (input @ up_weights) * gate_output
    # Need mm_out_tensor as intermediate
    mm_out = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_output_total_n], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_out_mem_config,
    )

    activated_output = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_output_total_n], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_out_mem_config,
    )

    activated_output = DRAMStreamingExpertsMatmul.op(
        input_a=x_replicated,
        input_b=up_expert_tensors[0],
        output_tensor=activated_output,
        fp32_dest_acc_en=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        index_tensor=expert_indices,
        selected_experts_k=selected_experts_k,
        subblock_k=subblock_k,
        mul_tensor=gate_output,
        mm_out_tensor=mm_out,
    )

    ttnn.deallocate(gate_output)
    ttnn.deallocate(mm_out)
    ttnn.deallocate(x_replicated)

    # ===== Down proj: per-expert computation =====
    # activated_output is [M, N_gate * selected_experts_k] on compute cores
    # For down_proj, input differs per expert, so we need per-expert calls
    # Reshape activated to separate experts: [selected_experts_k, M, N_gate]

    down_K_tiles = intermediate_size // TILE_SIZE
    down_subblock_k = down_K_tiles if down_K_tiles <= 8 else down_K_tiles // 2

    if selected_experts_k == 1:
        # Single expert: direct computation
        # Prepare input for down_proj
        in0_down_shard_spec = ttnn.ShardSpec(
            compute_core_grid, [M, intermediate_size], ttnn.ShardOrientation.ROW_MAJOR
        )
        in0_down_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_down_shard_spec
        )

        # Convert activated to HEIGHT_SHARDED for down_proj input
        activated_hs = ttnn.to_memory_config(activated_output, in0_down_mem_config)
        ttnn.deallocate(activated_output)

        # Down proj output
        down_out_shard_spec = ttnn.ShardSpec(
            compute_core_grid, [M, per_core_N_down], ttnn.ShardOrientation.ROW_MAJOR
        )
        down_out_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, down_out_shard_spec
        )
        down_output = ttnn.from_torch(
            torch.zeros([1, 1, M, N_down], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=down_out_mem_config,
        )

        down_output = DRAMStreamingExpertsMatmul.op(
            input_a=activated_hs,
            input_b=down_expert_tensors[0],
            output_tensor=down_output,
            fp32_dest_acc_en=False,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            index_tensor=expert_indices,
            selected_experts_k=1,
            subblock_k=down_subblock_k,
        )
        ttnn.deallocate(activated_hs)

        # Convert from WIDTH_SHARDED to interleaved for downstream ops
        down_output_interleaved = ttnn.to_memory_config(down_output, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(down_output)

        # Reshape to [1, num_experts_per_device, M, hidden_size] with zeros for non-selected experts
        # For now, place at position 0 and let combine handle routing
        # TODO: Place at correct expert position based on expert_indices
        result = ttnn.reshape(down_output_interleaved, [1, 1, M, hidden_size])

        # Pad expert dimension
        # For selected_experts_k=1, we need [1, num_experts_per_device, M, hidden] but only 1 filled
        # The combine expects this shape with expert outputs at the correct positions
        # For now, repeat to fill all expert slots (combine will select the right one)
        result = ttnn.repeat(result, ttnn.Shape((1, num_experts_per_device, 1, 1)))

        return result
    else:
        # Multi-expert: fall back to per-expert down_proj calls
        # Convert activated from WIDTH_SHARDED [M, N*K_sel] to interleaved
        activated_interleaved = ttnn.to_memory_config(activated_output, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(activated_output)

        # activated shape: [1, 1, M, intermediate_size * selected_experts_k]
        # Reshape to [selected_experts_k, 1, M, intermediate_size]
        activated_reshaped = ttnn.reshape(
            activated_interleaved, [selected_experts_k, 1, M, intermediate_size]
        )
        ttnn.deallocate(activated_interleaved)

        down_outputs = []
        for e in range(selected_experts_k):
            # Slice this expert's activated output
            expert_activated = ttnn.slice(
                activated_reshaped,
                [e, 0, 0, 0],
                [e + 1, 1, M, intermediate_size],
            )
            expert_activated = ttnn.reshape(expert_activated, [1, 1, M, intermediate_size])

            # Prepare HEIGHT_SHARDED input
            in0_down_shard_spec = ttnn.ShardSpec(
                compute_core_grid, [M, intermediate_size], ttnn.ShardOrientation.ROW_MAJOR
            )
            in0_down_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_down_shard_spec
            )
            expert_activated_hs = ttnn.to_memory_config(expert_activated, in0_down_mem_config)
            ttnn.deallocate(expert_activated)

            # Per-expert index tensor
            e_index_data = torch.zeros(num_cores, 16, dtype=torch.int32)
            e_index_data[:, 0] = e
            e_index_data = e_index_data.to(torch.uint16)
            e_index_tile = ttnn.Tile([1, 16])
            e_index_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
            e_index_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, e_index_shard_spec
            )
            e_index = ttnn.from_torch(
                e_index_data,
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=e_index_mem_config,
                tile=e_index_tile,
            )

            # Down proj output
            down_out_shard_spec = ttnn.ShardSpec(
                compute_core_grid, [M, per_core_N_down], ttnn.ShardOrientation.ROW_MAJOR
            )
            down_out_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, down_out_shard_spec
            )
            down_out = ttnn.from_torch(
                torch.zeros([1, 1, M, N_down], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=down_out_mem_config,
            )

            down_out = DRAMStreamingExpertsMatmul.op(
                input_a=expert_activated_hs,
                input_b=down_expert_tensors[0],
                output_tensor=down_out,
                fp32_dest_acc_en=False,
                math_fidelity=ttnn.MathFidelity.HiFi2,
                index_tensor=e_index,
                selected_experts_k=1,
                subblock_k=down_subblock_k,
            )
            ttnn.deallocate(expert_activated_hs)
            ttnn.deallocate(e_index)

            # Convert to interleaved
            down_out_i = ttnn.to_memory_config(down_out, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(down_out)
            down_outputs.append(down_out_i)

        ttnn.deallocate(activated_reshaped)

        # Stack down outputs: [selected_experts_k, 1, M, hidden_size]
        # Then permute to [1, selected_experts_k, M, hidden_size]
        result = ttnn.concat(down_outputs, dim=0)
        for t in down_outputs:
            ttnn.deallocate(t)

        result = ttnn.permute(result, (1, 0, 2, 3))
        result = ttnn.reshape(result, [1, selected_experts_k, M, hidden_size])

        # If selected_experts_k < num_experts_per_device, pad the expert dimension
        if selected_experts_k < num_experts_per_device:
            # Pad with zeros along expert dimension
            padding = ttnn.from_torch(
                torch.zeros([1, num_experts_per_device - selected_experts_k, M, hidden_size], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            result = ttnn.concat([result, padding], dim=1)
            ttnn.deallocate(padding)

        return result

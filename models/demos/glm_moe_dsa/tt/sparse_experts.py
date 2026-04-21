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

# Tile size for in1 weights (always 32x32)
TILE_SIZE = 32

# Number of DRAM banks on Blackhole
BH_NUM_DRAM_BANKS = 8

# Tiny tile for decode (1 row × 32 cols)
DECODE_TILE = (1, 32)


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

    # Upload each expert sequentially to ensure contiguous DRAM allocation
    expert_tensors = []

    for expert_local_idx in range(num_experts_per_device):
        per_device_weights = []
        for device_id in range(num_devices):
            global_expert_id = device_id * num_experts_per_device + expert_local_idx
            w = expert_weights_torch[0, global_expert_id].clone()  # [K, N]

            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))

            w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), TILE_SIZE, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)
            per_device_weights.append(w_shuffled)

        stacked = torch.cat(per_device_weights, dim=0)  # [num_devices, 1, K, N_padded]

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

    IMPORTANT: This kernel requires tiny tiles (Tile([1,32])) for M=1 decode.
    The input x should have shape [1, 1, total_tokens, hidden_size] where
    total_tokens is the number of dispatched tokens (typically 1 at bs=1 decode).

    The kernel processes each token row independently through the expert MLP:
    gate = SiLU(token @ gate_weights[expert_idx])
    up = token @ up_weights[expert_idx]
    activated = gate * up
    down = activated @ down_weights[expert_idx]

    Args:
        x: Input tensor [1, 1, total_tokens, hidden_size] in TILE layout
        gate_expert_tensors: List of WIDTH_SHARDED gate_proj weight tensors
        up_expert_tensors: List of WIDTH_SHARDED up_proj weight tensors
        down_expert_tensors: List of WIDTH_SHARDED down_proj weight tensors
        num_experts_per_device: Number of local experts
        hidden_size: Model hidden size
        intermediate_size: Expert intermediate size
        device: TTNN device
        selected_experts_k: Number of experts to compute
        expert_indices: Optional index tensor [num_cores, 16] uint16

    Returns:
        Expert output [1, num_experts_per_device, total_tokens, hidden_size]
    """
    logger.warning("sparse_expert_forward: NOT YET INTEGRATED — falling back to message only")
    # TODO: Implement the full sparse forward path with M=1 tiny tiles
    # For now, this function is a placeholder. The actual integration needs:
    # 1. Reshape input from [1, 1, total_tokens, hidden] to per-token [1, 1, 1, hidden]
    # 2. Convert each token to HEIGHT_SHARDED with Tile([1, 32])
    # 3. Call DRAMStreamingExpertsMatmul for gate_proj (SiLU fused)
    # 4. Call DRAMStreamingExpertsMatmul for up_proj (mul with gate output)
    # 5. Convert activated output back to HEIGHT_SHARDED for down_proj
    # 6. Call DRAMStreamingExpertsMatmul for down_proj
    # 7. Reshape to [1, num_experts_per_device, total_tokens, hidden_size]
    raise NotImplementedError(
        "sparse_expert_forward not yet integrated. "
        "Kernel validated (ALL TESTS PASS) but MoE integration pending."
    )

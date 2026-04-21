# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import struct

import ttnn


def float_to_bfloat16_packed(value):
    """Convert float to packed bfloat16 (two copies in uint32)"""
    # Convert float32 to bytes
    float_bytes = struct.pack("f", value)
    # Extract upper 16 bits (bfloat16 is truncated float32)
    bf16_bytes = float_bytes[2:4]  # upper 16 bits in little-endian layout
    # Pack two copies into uint32 (little endian)
    packed = int.from_bytes(bf16_bytes + bf16_bytes, byteorder="little")
    return packed


def float_to_uint32(value):
    """Convert float to uint32"""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def merge_per_core_runtime_args(*groups):
    """
    Merge per-core runtime arg groups in-order with core-aware concatenation.

    Each group is a list of tuples: (core_coord, list[int]).
    If a core appears in multiple groups, args are concatenated in group order.
    """
    merged = []
    core_to_index = {}
    for group in groups:
        for core, args in group:
            key = (core.x, core.y)
            args_list = list(args)
            if key in core_to_index:
                idx = core_to_index[key]
                merged_core, merged_args = merged[idx]
                merged[idx] = (merged_core, merged_args + args_list)
            else:
                core_to_index[key] = len(merged)
                merged.append((core, args_list))
    return merged


def merge_kernel_defines(*define_groups):
    """
    Merge kernel defines in-order with key-aware deduplication.

    Each input is an iterable of (name, value) tuples. First occurrence preserves
    ordering; later occurrences override the value for that define name.
    """
    merged = {}
    ordered_names = []
    for group in define_groups:
        for name, value in group:
            if name not in merged:
                ordered_names.append(name)
            merged[name] = value
    return [(name, merged[name]) for name in ordered_names]


def fabric_config_enables_torus_x(fabric_config) -> bool:
    return fabric_config in (
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    )


def fabric_config_enables_torus_y(fabric_config) -> bool:
    return fabric_config in (
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    )


def generate_mm_weights(shape, dtype):
    import torch

    torch_mm_weights = (torch.randn(shape, dtype=torch.float32) / (shape[-2] ** 0.5)).to(dtype)
    return torch_mm_weights
    # TODO: Review the below, which should provide a similar result
    # torch_mm_weights = torch.empty(shape, dtype=dtype)
    # # This assumes that weights are already pre-transposed, so inner dimension is the first dimension
    # # fan_in assumes the inner dimension is the second dimension, which is why we pass a transposed view
    # # Alternatively, we could pass the original shape and use fan_out
    # torch.nn.init.kaiming_normal_(torch_mm_weights.T, mode="fan_in", nonlinearity="linear")
    # return torch_mm_weights


def shuffle_dram_tiles(tensor, tile_size, num_banks):
    """Reorder tiles within each DRAM bank shard from row-major to column-major.

    WIDTH_SHARDED DRAM layout stores tiles row-major, but the streaming
    matmul kernel expects K tiles contiguous for each N column. This
    function transposes the tile order within each shard so that the
    kernel can linearly read K tiles at a time.

    Args:
        tensor: [*, K, N] tensor (supports batch dimensions).
        tile_size: Tile dimension (square tiles assumed).
        num_banks: Number of DRAM banks (shards).

    Returns:
        Same-shape tensor with tiles rearranged per shard.
    """
    import torch

    orig_shape = tensor.shape
    K, N = orig_shape[-2], orig_shape[-1]

    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]

    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    shards = tensor.reshape(-1, K, per_N)

    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
    shuffled_tiles = tiles[:, source_idx, :, :]

    shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

    shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch_size, K, n_padded)

    if needs_padding:
        shuffled = shuffled[:, :, :N]

    return shuffled.reshape(*orig_shape)


def pad_n_to_dram_banks(n, tile_size, num_banks):
    """Pad N dimension to align with DRAM bank sharding.

    Returns N_padded such that N_padded is divisible by (tile_size * num_banks).
    """
    lcm = tile_size * num_banks
    return ((n + lcm - 1) // lcm) * lcm


# Hardcoded optimal DRAM bank to logical worker assignment for Blackhole to avoid differences from harvesting
def get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, noc):
    import ttnn

    assert noc == ttnn.NOC.NOC_0, "Only NOC_0 is supported for now"
    assert device.arch() == ttnn.Arch.BLACKHOLE, "Only Blackhole is supported for now"
    return [
        ttnn.CoreCoord(0, 9),
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(0, 7),
        ttnn.CoreCoord(0, 3),
        ttnn.CoreCoord(7, 9),
        ttnn.CoreCoord(7, 1),
        ttnn.CoreCoord(7, 6),
        ttnn.CoreCoord(7, 4),
    ]

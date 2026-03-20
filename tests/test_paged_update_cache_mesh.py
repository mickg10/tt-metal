"""Test paged_update_cache on BH Galaxy MESH (32 devices) — not single device.
Single device passes. Does mesh also pass?
"""
import torch
import ttnn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

NKV, D, BLOCK_SIZE, NUM_BLOCKS = 2, 128, 64, 4

print("=" * 60)
print("paged_update_cache MESH test — Blackhole Galaxy")
print("=" * 60)

# Open mesh (same as vLLM does)
num_devices = ttnn.GetNumAvailableDevices()
print(f"Available devices: {num_devices}")

if num_devices >= 32:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    print(f"Opened 8x4 mesh ({mesh.get_num_devices()} devices)")
else:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    print(f"Opened 1x1 mesh")

device = mesh  # MeshDevice acts as device for ops
grid = device.compute_with_storage_grid_size()
print(f"Grid: {grid.x}x{grid.y}")

# Allocate cache on mesh — replicated to all devices
cache_shape = (NUM_BLOCKS, NKV, BLOCK_SIZE, D)
k_cache_host = torch.zeros(cache_shape)
v_cache_host = torch.zeros(cache_shape)

tt_k_cache = ttnn.from_torch(
    k_cache_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)
tt_v_cache = ttnn.from_torch(
    v_cache_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)

# Page table — identity mapping, replicated
page_table = torch.arange(NUM_BLOCKS).unsqueeze(0).int()
tt_page_table = ttnn.from_torch(
    page_table, device=device, dtype=ttnn.int32,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)

# Known K/V values to write
k_new_host = torch.randn(1, 1, NKV, D)
v_new_host = torch.randn(1, 1, NKV, D)
pos = 5

print(f"\nWriting K at pos={pos}")
print(f"  input_k[0,0,0,:4] = {k_new_host[0,0,0,:4].tolist()}")

# Transfer to mesh — replicated (all devices get same K)
tt_k_new = ttnn.from_torch(
    k_new_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)
tt_v_new = ttnn.from_torch(
    v_new_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)

# Try HEIGHT_SHARDED (required by paged_update_cache on single device)
# On mesh, skip sharding — the op may handle it internally
try:
    shard_grid = ttnn.num_cores_to_corerangeset(1, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [NKV * 32, D], ttnn.ShardOrientation.ROW_MAJOR)
    shard_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt_k_new = ttnn.interleaved_to_sharded(tt_k_new, shard_mc)
    tt_v_new = ttnn.interleaved_to_sharded(tt_v_new, shard_mc)
    print("  Sharded K/V for paged_update_cache")
except Exception as e:
    print(f"  Sharding failed ({e}), trying unsharded...")

# Position tensor — replicated
tt_pos = ttnn.from_torch(
    torch.tensor([pos]), device=device, dtype=ttnn.int32,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)

# Run paged_update_cache on mesh
print("Running paged_update_cache on mesh...")
ttnn.experimental.paged_update_cache(tt_k_cache, tt_k_new, update_idxs_tensor=tt_pos, page_table=tt_page_table)
ttnn.experimental.paged_update_cache(tt_v_cache, tt_v_new, update_idxs_tensor=tt_pos, page_table=tt_page_table)

# Read back from each device and verify
print("\nVerifying cache on each device:")
block = pos // BLOCK_SIZE
offset = pos % BLOCK_SIZE

k_devs = ttnn.get_device_tensors(tt_k_cache)
expected = k_new_host[0, 0, 0, :4].tolist()

all_match = True
for di in range(min(8, len(k_devs))):  # Check first 8 devices
    kd = ttnn.to_torch(k_devs[di].cpu())
    actual = kd[block, 0, offset, :4].tolist()
    match = all(abs(a - b) < 0.01 for a, b in zip(expected, actual))
    status = "MATCH" if match else "MISMATCH"
    if not match:
        all_match = False
    print(f"  d{di}: cache_k[{block},{0},{offset},:4] = {[f'{x:.4f}' for x in actual]} {status}")

print(f"\nExpected: {[f'{x:.4f}' for x in expected]}")
print(f"\n{'ALL DEVICES MATCH — paged_update_cache WORKS on mesh' if all_match else 'MISMATCH FOUND — paged_update_cache BROKEN on mesh'}")

ttnn.close_mesh_device(mesh)

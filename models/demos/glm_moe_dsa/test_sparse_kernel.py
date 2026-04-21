#!/usr/bin/env python3
"""Test DRAMStreamingExpertsMatmul kernel on Blackhole single device.

Validates:
1. WIDTH_SHARDED weight upload with tile shuffle
2. Expert indexing (select expert N out of M)
3. Correctness vs PyTorch reference
4. GLM-5.1 expert dimensions (K=6144, N=2048 for gate/up; K=2048, N=6144 for down)
"""

import os
import sys
import torch
import ttnn
from loguru import logger

from models.demos.glm_moe_dsa.b1_utils import shuffle_dram_tiles, pad_n_to_dram_banks
from models.demos.glm_moe_dsa.micro_ops.dram_streaming_experts_matmul.op import DRAMStreamingExpertsMatmul


def test_sparse_matmul(device, M, K, N, num_experts, selected_expert, dtype_b=ttnn.bfloat16, fused_silu=False):
    """Test DRAMStreamingExpertsMatmul with given dimensions."""
    num_banks = device.dram_grid_size().x
    tile_size = 32
    N_padded = pad_n_to_dram_banks(N, tile_size, num_banks)
    per_core_N = N_padded // num_banks

    logger.info(f"Test: M={M}, K={K}, N={N}, N_padded={N_padded}, per_core_N={per_core_N}, "
                f"experts={num_experts}, selected={selected_expert}, silu={fused_silu}")

    # Compute cores — use PINNED cores (same as kernel) for BH Galaxy mesh compatibility
    from models.demos.glm_moe_dsa.b1_utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
    all_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    num_cores = len(all_worker_cores)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
    )

    # ===== Input A: HEIGHT_SHARDED replicated =====
    torch.manual_seed(42)
    torch_in0 = torch.randn(1, 1, M, K).bfloat16().float()
    in0_replicated = torch_in0.repeat(1, 1, num_cores, 1)  # [1, 1, M*cores, K]
    in0_spec = ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_spec)
    in0_t = ttnn.from_torch(
        in0_replicated.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=in0_mem, tile=ttnn.Tile([32, 32]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # ===== Expert weights: WIDTH_SHARDED per expert =====
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))}
    )
    in1_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_spec)

    expert_tensors = []
    expert_weights_torch = []
    for e in range(num_experts):
        torch.manual_seed(100 + e)
        w = torch.randn(1, 1, K, N).bfloat16().float()
        expert_weights_torch.append(w)

        # Pad (skip shuffle to debug - kernel reads tiles as-is from DRAM)
        w_padded = w.clone()
        if N_padded != N:
            w_padded = torch.nn.functional.pad(w_padded, (0, N_padded - N))
        # DEBUGGING: Try both shuffled and unshuffled to see which gives better PCC
        use_shuffle = os.environ.get("SHUFFLE", "1") == "1"
        if use_shuffle:
            w_shuffled = shuffle_dram_tiles(w_padded.reshape(1, K, N_padded), tile_size, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)
        else:
            w_shuffled = w_padded  # No shuffle

        et = ttnn.from_torch(
            w_shuffled.bfloat16(), dtype=dtype_b, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=in1_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        expert_tensors.append(et)

    # ===== Index tensor =====
    idx = torch.zeros(num_cores, 16, dtype=torch.int32)
    idx[:, 0] = selected_expert
    idx = idx.to(torch.uint16)
    idx_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    idx_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, idx_spec)
    idx_t = ttnn.from_torch(
        idx, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=idx_mem, tile=ttnn.Tile([1, 16]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # ===== Output tensor =====
    out_spec = ttnn.ShardSpec(compute_core_grid, [M, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_spec)
    out_t = ttnn.from_torch(
        torch.zeros(1, 1, M, N_padded).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=out_mem, tile=ttnn.Tile([32, 32]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # ===== Run kernel =====
    Kt = K // tile_size
    subblock_k = Kt if Kt <= 8 else Kt // 2
    # Make sure subblock_k divides Kt
    while Kt % subblock_k != 0:
        subblock_k -= 1

    # Working buffer for CB1 (required by kernel for address wrapping)
    in1_tile_obj = ttnn.Tile([tile_size, tile_size])
    in1_tile_size_bytes = in1_tile_obj.get_tile_size(dtype_b)
    num_in1_buffers = 3
    in1_CB_tiles = subblock_k * num_in1_buffers
    wb_shard_shape = (tile_size, in1_CB_tiles * tile_size)
    wb_total_width = in1_CB_tiles * tile_size * num_cores
    wb_shard_spec = ttnn.ShardSpec(compute_core_grid, wb_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    wb_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, wb_shard_spec)
    wb_t = ttnn.from_torch(
        torch.zeros([1, 1, tile_size, wb_total_width]).bfloat16(),
        dtype=dtype_b, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=wb_mem, tile=in1_tile_obj,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    result = DRAMStreamingExpertsMatmul.op(
        input_a=in0_t,
        input_b=expert_tensors[0],  # Base address
        output_tensor=out_t,
        fp32_dest_acc_en=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        index_tensor=idx_t,
        selected_experts_k=1,
        subblock_k=subblock_k,
        fused_activation="silu" if fused_silu else None,
        working_buf_tensor=wb_t,
    )

    # ===== Verify against PyTorch reference =====
    # Get per-device tensors and extract device 0's output directly
    device_tensors = ttnn.get_device_tensors(result)
    result_torch = ttnn.to_torch(device_tensors[0])

    # PyTorch reference
    ref = torch_in0 @ expert_weights_torch[selected_expert]
    if fused_silu:
        ref = torch.nn.functional.silu(ref)

    # Compare (only up to N, ignore padding)
    result_np = result_torch[..., :N].float()
    ref_np = ref[..., :N].float()

    # PCC (Pearson Correlation Coefficient)
    r_flat = result_np.flatten()
    ref_flat = ref_np.flatten()
    pcc = torch.corrcoef(torch.stack([r_flat, ref_flat]))[0, 1].item()

    # Relative error
    rel_err = (result_np - ref_np).abs() / (ref_np.abs() + 1e-6)
    mean_rel_err = rel_err.mean().item()

    logger.info(f"  PCC: {pcc:.6f}, Mean Rel Error: {mean_rel_err:.6f}")
    logger.info(f"  Output: mean={result_np.mean():.4f}, std={result_np.std():.4f}")
    logger.info(f"  Ref:    mean={ref_np.mean():.4f}, std={ref_np.std():.4f}")

    # Cleanup
    for et in expert_tensors:
        ttnn.deallocate(et)
    ttnn.deallocate(in0_t)
    ttnn.deallocate(idx_t)
    ttnn.deallocate(result)
    ttnn.deallocate(wb_t)

    passed = pcc > 0.95
    logger.info(f"  {'PASS' if passed else 'FAIL'}: PCC={pcc:.4f} (threshold 0.95)")
    return passed


def main():
    logger.info("Opening BH Galaxy mesh (8, 4)...")
    # BH Galaxy needs fabric setup before opening mesh
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT)
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(8, 4),
        dispatch_core_config=ttnn.DispatchCoreConfig(),  # Default axis (COL for BH)
    )
    logger.info(f"Device opened: num_devices={device.get_num_devices()}, DRAM banks={device.dram_grid_size().x}")

    all_pass = True

    # Test 1: Small dimensions (quick sanity)
    # Test -1: Verify WIDTH_SHARDED upload/download works
    logger.info("=" * 60)
    logger.info("Test -1: WIDTH_SHARDED DRAM upload/download sanity check")
    num_banks_test = device.dram_grid_size().x
    K_t, N_t = 64, 256
    per_core_N_t = N_t // num_banks_test
    dram_grid_t = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks_test - 1, 0))})
    ws_spec = ttnn.ShardSpec(dram_grid_t, [K_t, per_core_N_t], ttnn.ShardOrientation.ROW_MAJOR)
    ws_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, ws_spec)
    torch.manual_seed(999)
    ws_data = torch.randn(1, 1, K_t, N_t).bfloat16()
    ws_tt = ttnn.from_torch(ws_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ws_mem, mesh_mapper=ttnn.ReplicateTensorToMesh(device))
    ws_dev0 = ttnn.get_device_tensors(ws_tt)
    ws_back = ttnn.to_torch(ws_dev0[0])
    ws_pcc = torch.corrcoef(torch.stack([ws_back.float().flatten(), ws_data.float().flatten()]))[0, 1].item()
    logger.info(f"  WIDTH_SHARDED upload/download PCC: {ws_pcc:.6f}")
    logger.info(f"  orig shape={ws_data.shape}, back shape={ws_back.shape}")
    ttnn.deallocate(ws_tt)
    all_pass &= ws_pcc > 0.99

    # Test 0: Verify standard ttnn.matmul works (sanity check for tensor extraction)
    logger.info("=" * 60)
    logger.info("Test 0: Standard ttnn.matmul sanity check (M=32, K=64, N=256)")
    torch.manual_seed(42)
    t_a = torch.randn(1, 1, 32, 64).bfloat16()
    torch.manual_seed(100)
    t_b = torch.randn(1, 1, 64, 256).bfloat16()
    ref_0 = t_a.float() @ t_b.float()
    tt_a = ttnn.from_torch(t_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(device))
    tt_b = ttnn.from_torch(t_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(device))
    tt_c = ttnn.matmul(tt_a, tt_b)
    dev_tensors = ttnn.get_device_tensors(tt_c)
    tt_c_torch = ttnn.to_torch(dev_tensors[0])
    pcc_0 = torch.corrcoef(torch.stack([tt_c_torch.float().flatten(), ref_0.flatten()]))[0, 1].item()
    logger.info(f"  Standard matmul PCC: {pcc_0:.6f}")
    ttnn.deallocate(tt_a)
    ttnn.deallocate(tt_b)
    ttnn.deallocate(tt_c)
    all_pass &= pcc_0 > 0.99

    # Test 1a: WITHOUT indexing (expert 0 only, no index tensor) to isolate tile shuffle
    logger.info("=" * 60)
    logger.info("Test 1a: Small dims NO INDEX (M=32, K=64, N=256, 1 expert)")
    all_pass &= test_sparse_matmul(device, M=32, K=64, N=256, num_experts=1, selected_expert=0)

    # Test 1b: With indexing, expert 0 (should match expert 0)
    logger.info("=" * 60)
    logger.info("Test 1b: Small dims INDEX=0 (M=32, K=64, N=256, 4 experts)")
    all_pass &= test_sparse_matmul(device, M=32, K=64, N=256, num_experts=4, selected_expert=0)

    # Test 2: Expert 2 selection
    logger.info("=" * 60)
    logger.info("Test 2: Small dims INDEX=2 (M=32, K=64, N=256, 4 experts)")
    all_pass &= test_sparse_matmul(device, M=32, K=64, N=256, num_experts=4, selected_expert=2)

    # Test 3: GLM-5.1 gate/up dimensions (K=6144, N=2048) - expert 0 no index
    logger.info("=" * 60)
    logger.info("Test 3: GLM-5.1 gate/up dims NO INDEX (M=32, K=6144, N=2048, 1 expert)")
    all_pass &= test_sparse_matmul(device, M=32, K=6144, N=2048, num_experts=1, selected_expert=0,
                                    dtype_b=ttnn.bfloat4_b)

    # Test 4: GLM-5.1 down dimensions (K=2048, N=6144) - expert 0 no index
    logger.info("=" * 60)
    logger.info("Test 4: GLM-5.1 down dims NO INDEX (M=32, K=2048, N=6144, 1 expert)")
    all_pass &= test_sparse_matmul(device, M=32, K=2048, N=6144, num_experts=1, selected_expert=0,
                                    dtype_b=ttnn.bfloat8_b)

    logger.info("=" * 60)
    if all_pass:
        logger.info("ALL TESTS PASSED!")
    else:
        logger.error("SOME TESTS FAILED!")

    ttnn.close_mesh_device(device)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test DRAMStreamingExpertsMatmul kernel on Blackhole single device.

Validates:
1. WIDTH_SHARDED weight upload with tile shuffle
2. Expert indexing (select expert N out of M)
3. Correctness vs PyTorch reference
4. GLM-5.1 expert dimensions (K=6144, N=2048 for gate/up; K=2048, N=6144 for down)
"""

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

    # Compute cores
    all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
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

        # Pad and shuffle
        w_padded = w.clone()
        if N_padded != N:
            w_padded = torch.nn.functional.pad(w_padded, (0, N_padded - N))
        w_shuffled = shuffle_dram_tiles(w_padded.reshape(1, K, N_padded), tile_size, num_banks)
        w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)

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
    )

    # ===== Verify against PyTorch reference =====
    # Convert WIDTH_SHARDED to interleaved, then extract from first device
    result_interleaved = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
    # Get per-device tensors, take device 0
    device_tensors = ttnn.get_device_tensors(result_interleaved)
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

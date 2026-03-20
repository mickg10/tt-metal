"""
Comprehensive paged_update_cache correctness tests for Blackhole Galaxy.

BUG: paged_update_cache writes WRONG VALUES to KV cache on Blackhole hardware.
The kernel uses pack_untilize_block -> modify single row -> tilize_block pipeline
which is broken on BH (upstream Issue #14594). paged_fill_cache (full tile writes)
works fine because it bypasses the untilize/tilize path entirely.

This test file is designed to:
  - Run as a standalone script (no pytest required -- BH Galaxy may lack packages)
  - Also run under pytest when available
  - FAIL on current BH until the untilize/tilize fix is applied
  - PASS on Wormhole (regression protection)

Usage:
  # Standalone (inside container on BH Galaxy):
  python tests/test_paged_update_cache_bh.py

  # With pytest (on CI or dev machines):
  TT_ENABLE_HW_TESTS=1 pytest tests/test_paged_update_cache_bh.py -v

Hardware: Blackhole (13x10 = 130 cores), also works on Wormhole (8x8 = 64 cores).
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Lazy import ttnn -- allows reading this file without a device present.
# ---------------------------------------------------------------------------
try:
    import ttnn
except ImportError:
    print("ERROR: ttnn not importable. Run inside a tt-metal environment.")
    sys.exit(1)

# Try to import pytest; fall back to standalone mode if unavailable.
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE_HEIGHT = 32
TILE_WIDTH = 32
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def comp_pcc(golden: torch.Tensor, calculated: torch.Tensor, threshold: float = 0.99) -> Tuple[bool, float]:
    """Compute Pearson correlation coefficient between two tensors."""
    g = golden.float().flatten()
    c = calculated.float().flatten()
    if g.shape != c.shape:
        return False, 0.0
    if g.numel() == 0:
        return True, 1.0
    if torch.all(g == 0) and torch.all(c == 0):
        return True, 1.0
    pcc = torch.corrcoef(torch.stack([g, c]))[0, 1].item()
    if np.isnan(pcc):
        return False, 0.0
    return pcc >= threshold, pcc


def prepare_sharded_update(
    update_data: torch.Tensor,
    num_heads: int,
    head_dim: int,
    batch: int,
    device,
    grid,
) -> ttnn.Tensor:
    """
    Prepare a HEIGHT_SHARDED update tensor for paged_update_cache.

    Follows the exact pattern from the upstream nightly tests
    (test_paged_update_cache.py:run_test_update_cache_decode):
      1. Pad heads to nearest multiple of 32 (tile height)
      2. Convert to TILE_LAYOUT via ttnn.Tensor constructor
      3. Reshape back to logical shape (keeps padded backing)
      4. Compute shard dims from volume
      5. Move to HEIGHT_SHARDED L1

    Args:
        update_data: [1, batch, num_heads, head_dim] bf16 tensor
        num_heads: actual number of KV heads
        head_dim: head dimension
        batch: batch size (= num users)
        device: ttnn device
        grid: device compute grid
    Returns:
        HEIGHT_SHARDED ttnn tensor ready for paged_update_cache
    """
    input_shape = [1, batch, num_heads, head_dim]
    padded_heads = ((num_heads + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    x_pad = torch.nn.functional.pad(
        update_data.bfloat16(), (0, 0, 0, padded_heads - num_heads), "constant", 0
    )

    xt = ttnn.Tensor(x_pad, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    xt = ttnn.reshape(xt, ttnn.Shape(input_shape))

    num_cores = batch
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [
            xt.volume() // xt.padded_shape[-1] // num_cores,
            xt.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    xt = xt.to(device, mem_cfg)
    return xt


def torch_sdpa_ref(
    q: torch.Tensor,   # [1, B, NH, D]
    k: torch.Tensor,   # [B, NKV, S, D]
    v: torch.Tensor,   # [B, NKV, S, D]
    cur_pos: List[int],
    scale: float,
) -> torch.Tensor:
    """Torch reference SDPA decode with causal mask."""
    b = q.shape[1]
    nh = q.shape[2]
    nkv = k.shape[1]
    s = k.shape[2]
    d = q.shape[3]
    gqa = nh // nkv
    outputs = []
    for bi in range(b):
        pos = cur_pos[bi]
        mask = torch.full((1, s), float("-inf"))
        mask[0, : pos + 1] = 0.0
        for kvi in range(nkv):
            qi = q[0, bi, kvi * gqa : (kvi + 1) * gqa, :]
            ki = k[bi, kvi, :, :]
            vi = v[bi, kvi, :, :]
            sc = torch.matmul(qi, ki.T) * scale + mask
            w = torch.softmax(sc.float(), dim=-1).to(vi.dtype)
            outputs.append(torch.matmul(w, vi))
    result = torch.stack(outputs, dim=0).reshape(b, nh, d)
    return result.unsqueeze(0)


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    status: str  # PASS, FAIL, ERROR, SKIP
    detail: str = ""
    pcc: float = 0.0


ALL_RESULTS: List[TestResult] = []


def record(name: str, status: str, detail: str = "", pcc: float = 0.0):
    ALL_RESULTS.append(TestResult(name=name, status=status, detail=detail, pcc=pcc))
    flag = {"PASS": "OK", "FAIL": "XX", "ERROR": "!!", "SKIP": "--"}.get(status, "??")
    print(f"  [{flag}] {status}: {name}")
    if detail:
        print(f"         {detail}")


# ============================================================================
# TEST 1: Standalone paged_update_cache correctness
# ============================================================================
def test_1_single_row_update(device, grid):
    """
    TEST 1: Single-row paged_update_cache correctness.

    For each (position, dtype, num_heads, head_dim) combination:
      1. Create paged KV cache filled with known data via paged_fill_cache
      2. Write a single row via paged_update_cache at a specific position
      3. Read the cache back to host
      4. Verify the written row matches the input
      5. Verify the rest of the cache is unchanged

    Positions tested: 0, 31, 32, 63, 64 (tile boundaries)
    Dtypes: bf16, bf8
    Heads: 1, 2, 8
    Head dims: 64, 128
    """
    print("\n--- TEST 1: Single-row paged_update_cache correctness ---")
    torch.manual_seed(SEED)

    positions = [0, 31, 32, 63, 64]
    cache_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b]
    head_counts = [1, 2, 8]
    head_dims = [64, 128]

    batch = 1
    block_size = 64
    num_blocks = 4

    total = 0
    passed = 0

    for cache_dtype in cache_dtypes:
        for num_heads in head_counts:
            for head_dim in head_dims:
                for pos in positions:
                    test_name = f"1: pos={pos}, dtype={cache_dtype}, heads={num_heads}, dim={head_dim}"
                    total += 1
                    try:
                        cache_shape = (batch * num_blocks, num_heads, block_size, head_dim)
                        prefix_len = block_size * num_blocks

                        # Fill cache with known data via paged_fill_cache (proven working on BH)
                        prefix_data = torch.randn(1, num_heads, prefix_len, head_dim).bfloat16().float()

                        tt_cache = ttnn.from_torch(
                            torch.zeros(cache_shape).bfloat16(),
                            device=device, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT,
                        )
                        page_table = torch.arange(num_blocks).unsqueeze(0).int()
                        tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

                        fill_dtype = ttnn.bfloat16 if cache_dtype == ttnn.bfloat16 else cache_dtype
                        tt_prefix = ttnn.from_torch(
                            prefix_data.bfloat16(), device=device, dtype=fill_dtype, layout=ttnn.TILE_LAYOUT,
                        )
                        ttnn.experimental.paged_fill_cache(tt_cache, tt_prefix, tt_page_table, batch_idx=0)
                        ttnn.deallocate(tt_prefix)

                        baseline_cache = ttnn.to_torch(ttnn.from_device(tt_cache))

                        # Create update data and prepare sharded tensor
                        update_data = torch.randn(1, batch, num_heads, head_dim).bfloat16().float()
                        tt_update = prepare_sharded_update(update_data, num_heads, head_dim, batch, device, grid)

                        tt_pos = ttnn.from_torch(
                            torch.tensor([pos], dtype=torch.int32), device=device, dtype=ttnn.int32,
                        )
                        ttnn.experimental.paged_update_cache(
                            tt_cache, tt_update,
                            update_idxs_tensor=tt_pos, page_table=tt_page_table,
                        )

                        result_cache = ttnn.to_torch(ttnn.from_device(tt_cache))

                        # Check 1: Updated row matches input
                        block_idx = pos // block_size
                        row_in_block = pos % block_size
                        got_row = result_cache[block_idx, :num_heads, row_in_block : row_in_block + 1, :]
                        want_row = update_data[0, 0, :num_heads, :].unsqueeze(1)

                        if cache_dtype == ttnn.bfloat16:
                            row_ok = torch.equal(got_row, want_row.bfloat16())
                            row_detail = "exact" if row_ok else f"max_err={(got_row.float()-want_row.float()).abs().max().item():.8f}"
                        else:
                            row_ok, row_pcc = comp_pcc(want_row, got_row, threshold=0.98)
                            row_detail = f"PCC={row_pcc:.6f}"

                        # Check 2: Non-updated rows unchanged
                        baseline_check = baseline_cache.clone()
                        result_check = result_cache.clone()
                        baseline_check[block_idx, :, row_in_block, :] = 0.0
                        result_check[block_idx, :, row_in_block, :] = 0.0

                        if cache_dtype == ttnn.bfloat16:
                            rest_ok = torch.equal(baseline_check, result_check)
                        else:
                            rest_ok, rest_pcc = comp_pcc(baseline_check, result_check, threshold=0.999)

                        if row_ok and rest_ok:
                            record(test_name, "PASS", row_detail)
                            passed += 1
                        elif not row_ok:
                            record(test_name, "FAIL", f"Updated row mismatch: {row_detail}")
                        else:
                            record(test_name, "FAIL", "Non-updated rows corrupted")

                        ttnn.deallocate(tt_update)
                        ttnn.deallocate(tt_pos)
                        ttnn.deallocate(tt_cache)
                        ttnn.deallocate(tt_page_table)

                    except Exception as e:
                        record(test_name, "ERROR", f"{type(e).__name__}: {e}")
                        traceback.print_exc()

    print(f"\n  TEST 1 SUMMARY: {passed}/{total} passed")
    return passed == total


# ============================================================================
# TEST 2: Tile boundary stress test
# ============================================================================
def test_2_tile_boundary_stress(device, grid):
    """
    TEST 2: Tile boundary stress test.

    Tests EVERY position within the first block (0..63) to catch
    any position-dependent corruption in the untilize->modify->retilize pipeline.

    Uses bf16 cache + bf16 input for exact comparison (no quantization noise).
    """
    print("\n--- TEST 2: Tile boundary stress test (positions 0..63) ---")
    torch.manual_seed(SEED)

    batch = 1
    num_heads = 2
    head_dim = 128
    block_size = 64
    num_blocks = 2
    cache_dtype = ttnn.bfloat16

    cache_shape = (batch * num_blocks, num_heads, block_size, head_dim)
    page_table = torch.arange(num_blocks).unsqueeze(0).int()

    total = 0
    passed = 0
    failures = []

    for pos in range(64):
        test_name = f"2: tile_boundary pos={pos}"
        total += 1
        try:
            initial_data = torch.randn(cache_shape).bfloat16()
            tt_cache = ttnn.from_torch(initial_data, device=device, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT)
            tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

            torch.manual_seed(SEED + pos)
            update_data = torch.randn(1, batch, num_heads, head_dim).bfloat16().float()
            tt_update = prepare_sharded_update(update_data, num_heads, head_dim, batch, device, grid)

            tt_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device, dtype=ttnn.int32)

            ttnn.experimental.paged_update_cache(
                tt_cache, tt_update,
                update_idxs_tensor=tt_pos, page_table=tt_page_table,
            )

            result = ttnn.to_torch(ttnn.from_device(tt_cache))

            # Verify updated row
            block_idx = pos // block_size
            row_in_block = pos % block_size
            got = result[block_idx, :num_heads, row_in_block, :]
            want = update_data[0, 0, :num_heads, :].bfloat16()

            row_ok = torch.equal(got, want)

            # Verify non-updated rows unchanged
            rest_ok = True
            for blk in range(num_blocks):
                for row in range(block_size):
                    if blk == block_idx and row == row_in_block:
                        continue
                    if not torch.equal(result[blk, :num_heads, row, :], initial_data[blk, :num_heads, row, :]):
                        rest_ok = False
                        break
                if not rest_ok:
                    break

            if row_ok and rest_ok:
                passed += 1
            else:
                failures.append(pos)
                if not row_ok:
                    max_err = (got.float() - want.float()).abs().max().item()
                    record(test_name, "FAIL", f"row mismatch, max_err={max_err:.8f}")
                else:
                    record(test_name, "FAIL", "non-updated rows corrupted")

            ttnn.deallocate(tt_update)
            ttnn.deallocate(tt_pos)
            ttnn.deallocate(tt_cache)
            ttnn.deallocate(tt_page_table)

        except Exception as e:
            record(test_name, "ERROR", str(e))
            failures.append(pos)

    print(f"\n  TEST 2 SUMMARY: {passed}/{total} passed")
    if failures:
        print(f"  Failed positions: {failures}")
        boundary_fails = [p for p in failures if p in (0, 31, 32, 63)]
        if boundary_fails:
            print(f"  ** TILE BOUNDARY FAILURES: {boundary_fails} **")
    return passed == total


# ============================================================================
# TEST 3: Multi-step cache update (simulates actual decode)
# ============================================================================
def test_3_multi_step_decode(device, grid):
    """
    TEST 3: Multi-step sequential cache updates.

    Simulates actual decode:
      1. Fill cache with prefix (5 tokens) via paged_fill_cache
      2. For each of 10 decode steps:
         a. Write new K/V at next position via paged_update_cache
         b. Read back cache and verify ALL positions 0..current are correct
      3. This catches bugs where earlier writes get corrupted by later ones

    Tests with bf16 and bf8 cache dtypes.
    """
    print("\n--- TEST 3: Multi-step decode simulation ---")
    torch.manual_seed(SEED)

    all_passed = True

    for cache_dtype in [ttnn.bfloat16, ttnn.bfloat8_b]:
        dtype_name = "bf16" if cache_dtype == ttnn.bfloat16 else "bf8"
        test_name = f"3: multi_step_{dtype_name}"

        try:
            batch = 1
            num_heads = 2
            head_dim = 128
            block_size = 64
            num_blocks = 4
            prefix_len = 5
            num_decode_steps = 10

            cache_shape = (batch * num_blocks, num_heads, block_size, head_dim)
            page_table = torch.arange(num_blocks).unsqueeze(0).int()
            tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

            host_cache = torch.zeros(batch, num_heads, block_size * num_blocks, head_dim).bfloat16()

            tt_cache = ttnn.from_torch(
                torch.zeros(cache_shape).bfloat16(),
                device=device, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT,
            )

            # Fill prefix via paged_fill_cache
            prefix_data = torch.randn(1, num_heads, prefix_len, head_dim).bfloat16()
            padded_prefix_len = ((prefix_len + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
            prefix_padded = torch.zeros(1, num_heads, padded_prefix_len, head_dim).bfloat16()
            prefix_padded[:, :, :prefix_len, :] = prefix_data

            fill_dtype = ttnn.bfloat16 if cache_dtype == ttnn.bfloat16 else cache_dtype
            tt_prefix = ttnn.from_torch(prefix_padded, device=device, dtype=fill_dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.experimental.paged_fill_cache(tt_cache, tt_prefix, tt_page_table, batch_idx=0)
            ttnn.deallocate(tt_prefix)

            host_cache[0, :, :prefix_len, :] = prefix_data[0]

            step_passed = True
            for step in range(num_decode_steps):
                pos = prefix_len + step
                torch.manual_seed(SEED + 1000 + step)
                update_data = torch.randn(1, batch, num_heads, head_dim).bfloat16().float()

                host_cache[0, :, pos, :] = update_data[0, 0, :, :].bfloat16()

                tt_update = prepare_sharded_update(update_data, num_heads, head_dim, batch, device, grid)
                tt_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device, dtype=ttnn.int32)

                ttnn.experimental.paged_update_cache(
                    tt_cache, tt_update,
                    update_idxs_tensor=tt_pos, page_table=tt_page_table,
                )
                ttnn.deallocate(tt_update)
                ttnn.deallocate(tt_pos)

                # Read back and verify ALL positions 0..pos
                device_cache = ttnn.to_torch(ttnn.from_device(tt_cache))
                device_unpaged = (
                    device_cache
                    .reshape(batch, num_blocks, num_heads, block_size, head_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(batch, num_heads, block_size * num_blocks, head_dim)
                )

                for check_pos in range(pos + 1):
                    got = device_unpaged[0, :, check_pos, :]
                    want = host_cache[0, :, check_pos, :]
                    if cache_dtype == ttnn.bfloat16:
                        match = torch.equal(got, want)
                    else:
                        ok, pcc_val = comp_pcc(want, got, threshold=0.98)
                        match = ok

                    if not match:
                        err = (got.float() - want.float()).abs().max().item()
                        record(
                            f"{test_name} step={step} check_pos={check_pos}",
                            "FAIL",
                            f"Position {check_pos} corrupted after writing pos={pos}, max_err={err:.6f}",
                        )
                        step_passed = False
                        break

                if not step_passed:
                    break

            if step_passed:
                record(test_name, "PASS", f"All {num_decode_steps} steps verified")
            else:
                all_passed = False

            ttnn.deallocate(tt_cache)
            ttnn.deallocate(tt_page_table)

        except Exception as e:
            record(test_name, "ERROR", f"{type(e).__name__}: {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


# ============================================================================
# TEST 4: Cache update + SDPA decode integration test
# ============================================================================
def test_4_cache_update_sdpa_integration(device, grid):
    """
    TEST 4: paged_update_cache + paged_scaled_dot_product_attention_decode.

    End-to-end test matching the GLM model's exact flow:
      1. Create cache with GLM-like dimensions (NKV=2, D=128, block_size=64)
      2. Fill prefix via paged_fill_cache
      3. For each decode step:
         a. Update cache via paged_update_cache (with HEIGHT_SHARDED input)
         b. Run paged_scaled_dot_product_attention_decode
         c. Compare output with torch reference
      4. PCC threshold: 0.98
    """
    print("\n--- TEST 4: Cache update + SDPA decode integration ---")
    torch.manual_seed(SEED)

    B = 1
    NH = 24   # Q heads
    NKV = 2   # KV heads
    D = 128   # head dim
    BLOCK_SIZE = 64
    NUM_BLOCKS = 4
    PREFIX_LEN = 5
    NUM_DECODE_STEPS = 5
    SCALE = D ** -0.5

    try:
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(min(8, grid.x), min(8, grid.y)),
            q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False,
        )
        cc_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False, packer_l1_acc=False,
        )

        cache_shape = (B * NUM_BLOCKS, NKV, BLOCK_SIZE, D)
        tt_k_cache = ttnn.from_torch(
            torch.zeros(cache_shape).bfloat16(), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        tt_v_cache = ttnn.from_torch(
            torch.zeros(cache_shape).bfloat16(), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        page_table = torch.arange(NUM_BLOCKS).unsqueeze(0).int()
        tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

        k_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
        v_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)

        # Fill prefix
        k_prefix = torch.randn(1, NKV, PREFIX_LEN, D).bfloat16().float()
        v_prefix = torch.randn(1, NKV, PREFIX_LEN, D).bfloat16().float()
        padded_prefix_len = ((PREFIX_LEN + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
        k_prefix_padded = torch.zeros(1, NKV, padded_prefix_len, D).bfloat16()
        v_prefix_padded = torch.zeros(1, NKV, padded_prefix_len, D).bfloat16()
        k_prefix_padded[:, :, :PREFIX_LEN, :] = k_prefix.bfloat16()
        v_prefix_padded[:, :, :PREFIX_LEN, :] = v_prefix.bfloat16()

        tt_k_prefix = ttnn.from_torch(k_prefix_padded, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tt_v_prefix = ttnn.from_torch(v_prefix_padded, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.experimental.paged_fill_cache(tt_k_cache, tt_k_prefix, tt_page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(tt_v_cache, tt_v_prefix, tt_page_table, batch_idx=0)
        ttnn.deallocate(tt_k_prefix)
        ttnn.deallocate(tt_v_prefix)

        k_full[0, :, :PREFIX_LEN, :] = k_prefix[0]
        v_full[0, :, :PREFIX_LEN, :] = v_prefix[0]

        all_passed = True
        for step in range(NUM_DECODE_STEPS):
            pos = PREFIX_LEN + step
            torch.manual_seed(SEED + 2000 + step)

            k_new = torch.randn(1, B, NKV, D).bfloat16().float()
            v_new = torch.randn(1, B, NKV, D).bfloat16().float()
            q_decode = torch.randn(1, B, NH, D).bfloat16().float()

            k_full[0, :, pos, :] = k_new[0, 0, :, :]
            v_full[0, :, pos, :] = v_new[0, 0, :, :]

            tt_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device, dtype=ttnn.int32)

            tt_k_new = prepare_sharded_update(k_new, NKV, D, B, device, grid)
            tt_v_new = prepare_sharded_update(v_new, NKV, D, B, device, grid)

            ttnn.experimental.paged_update_cache(
                tt_k_cache, tt_k_new,
                update_idxs_tensor=tt_pos, page_table=tt_page_table,
            )
            ttnn.experimental.paged_update_cache(
                tt_v_cache, tt_v_new,
                update_idxs_tensor=tt_pos, page_table=tt_page_table,
            )
            ttnn.deallocate(tt_k_new)
            ttnn.deallocate(tt_v_new)

            # SDPA decode
            tt_q = ttnn.from_torch(q_decode.bfloat16(), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_q, tt_k_cache, tt_v_cache,
                page_table_tensor=tt_page_table,
                cur_pos_tensor=tt_pos,
                scale=SCALE,
                program_config=sdpa_cfg,
                compute_kernel_config=cc_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            tt_result = ttnn.to_torch(tt_out)[..., :NH, :D]
            ref = torch_sdpa_ref(q_decode, k_full, v_full, [pos], SCALE)

            ok, pcc_val = comp_pcc(ref, tt_result, threshold=0.98)
            max_err = (ref - tt_result.float()).abs().max().item()

            step_name = f"4: SDPA step={step} pos={pos}"
            if ok:
                record(step_name, "PASS", f"PCC={pcc_val:.6f}, max_err={max_err:.6f}", pcc_val)
            else:
                record(step_name, "FAIL", f"PCC={pcc_val:.6f}, max_err={max_err:.6f}", pcc_val)
                all_passed = False

            ttnn.deallocate(tt_q)
            ttnn.deallocate(tt_out)
            ttnn.deallocate(tt_pos)

        ttnn.deallocate(tt_k_cache)
        ttnn.deallocate(tt_v_cache)
        ttnn.deallocate(tt_page_table)

        return all_passed

    except Exception as e:
        record("4: SDPA integration", "ERROR", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Multi-user paged_update_cache (batch > 1)
# ============================================================================
def test_5_multi_user_update(device, grid):
    """
    TEST 5: Multi-user paged_update_cache.

    Verifies that when batch > 1, each user's cache is updated independently
    and no cross-user corruption occurs. Uses shuffled page tables.
    """
    print("\n--- TEST 5: Multi-user paged_update_cache ---")
    torch.manual_seed(SEED)

    batch = 4
    num_heads = 2
    head_dim = 128
    block_size = 64
    max_seq_len = 256
    blocks_per_user = max_seq_len // block_size
    num_blocks = batch * blocks_per_user

    try:
        permutation = torch.randperm(num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch, blocks_per_user)

        initial_data = torch.randn(num_blocks, num_heads, block_size, head_dim).bfloat16()
        shuffled_data = initial_data[permutation]

        unpaged = (
            initial_data
            .reshape(batch, blocks_per_user, num_heads, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, num_heads, max_seq_len, head_dim)
        )

        tt_cache = ttnn.from_torch(shuffled_data, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

        update_positions = [10, 31, 32, 63]  # one per user, hits tile boundaries
        update_data = torch.randn(1, batch, num_heads, head_dim).bfloat16().float()

        tt_update = prepare_sharded_update(update_data, num_heads, head_dim, batch, device, grid)

        tt_pos = ttnn.from_torch(
            torch.tensor(update_positions, dtype=torch.int32), device=device, dtype=ttnn.int32,
        )

        ttnn.experimental.paged_update_cache(
            tt_cache, tt_update,
            update_idxs_tensor=tt_pos, page_table=tt_page_table,
        )

        result_shuffled = ttnn.to_torch(ttnn.from_device(tt_cache))
        result_unshuffled = result_shuffled[reverse_permutation]
        result_unpaged = (
            result_unshuffled
            .reshape(batch, blocks_per_user, num_heads, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, num_heads, max_seq_len, head_dim)
        )

        expected = unpaged.clone()
        for i in range(batch):
            pos = update_positions[i]
            expected[i, :num_heads, pos, :] = update_data[0, i, :num_heads, :].bfloat16()

        exact_match = torch.equal(result_unpaged, expected)
        if exact_match:
            record("5: multi_user", "PASS", "Exact match with shuffled page table")
        else:
            max_err = (result_unpaged.float() - expected.float()).abs().max().item()
            for i in range(batch):
                pos = update_positions[i]
                user_row = result_unpaged[i, :num_heads, pos, :]
                want_row = expected[i, :num_heads, pos, :]
                if not torch.equal(user_row, want_row):
                    user_err = (user_row.float() - want_row.float()).abs().max().item()
                    record(f"5: multi_user user={i} pos={pos}", "FAIL",
                           f"Updated row mismatch, max_err={user_err:.8f}")
            record("5: multi_user", "FAIL", f"max_err={max_err:.8f}")

        ttnn.deallocate(tt_update)
        ttnn.deallocate(tt_pos)
        ttnn.deallocate(tt_cache)
        ttnn.deallocate(tt_page_table)

        return exact_match

    except Exception as e:
        record("5: multi_user", "ERROR", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# TEST 6: Known-value canary test
# ============================================================================
def test_6_canary_values(device, grid):
    """
    TEST 6: Known-value canary test.

    Fills cache with zeros, writes known constant rows (1.0, 2.0, 3.0, etc.)
    at specific positions, then reads back and checks for exact values.
    This makes debugging easier -- if the kernel writes wrong values,
    the expected pattern makes it obvious what went wrong.
    """
    print("\n--- TEST 6: Known-value canary test ---")
    torch.manual_seed(SEED)

    batch = 1
    num_heads = 1
    head_dim = 128
    block_size = 64
    num_blocks = 2

    cache_shape = (batch * num_blocks, num_heads, block_size, head_dim)
    page_table = torch.arange(num_blocks).unsqueeze(0).int()

    try:
        tt_cache = ttnn.from_torch(
            torch.zeros(cache_shape).bfloat16(),
            device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

        canary_positions = [0, 15, 31, 32, 47, 63]
        canary_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        all_passed = True
        for pos, val in zip(canary_positions, canary_values):
            update_data = torch.full((1, batch, num_heads, head_dim), val).bfloat16().float()
            tt_update = prepare_sharded_update(update_data, num_heads, head_dim, batch, device, grid)

            tt_pos = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32), device=device, dtype=ttnn.int32
            )

            ttnn.experimental.paged_update_cache(
                tt_cache, tt_update,
                update_idxs_tensor=tt_pos, page_table=tt_page_table,
            )
            ttnn.deallocate(tt_update)
            ttnn.deallocate(tt_pos)

        result = ttnn.to_torch(ttnn.from_device(tt_cache))
        for pos, val in zip(canary_positions, canary_values):
            block_idx = pos // block_size
            row_in_block = pos % block_size
            got_row = result[block_idx, 0, row_in_block, :]
            expected_val = torch.tensor(val).bfloat16().item()

            all_canary = torch.all(got_row == expected_val).item()

            test_name = f"6: canary pos={pos} val={val}"
            if all_canary:
                record(test_name, "PASS")
            else:
                actual_mean = got_row.float().mean().item()
                actual_std = got_row.float().std().item()
                record(test_name, "FAIL",
                       f"Expected all {expected_val}, got mean={actual_mean:.4f} std={actual_std:.4f}")
                all_passed = False

        # Check non-canary positions are still zero
        for block_idx in range(num_blocks):
            for row in range(block_size):
                global_pos = block_idx * block_size + row
                if global_pos in canary_positions:
                    continue
                got = result[block_idx, 0, row, :]
                if not torch.all(got == 0).item():
                    non_zero_count = (got != 0).sum().item()
                    record(f"6: canary zero_check pos={global_pos}", "FAIL",
                           f"{non_zero_count}/{head_dim} elements non-zero")
                    all_passed = False
                    break

        if all_passed:
            record("6: canary overall", "PASS", "All canary values correct, zeros preserved")

        ttnn.deallocate(tt_cache)
        ttnn.deallocate(tt_page_table)

        return all_passed

    except Exception as e:
        record("6: canary", "ERROR", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# TEST 7: GLM-4.7 exact dimensions (regression test for the model)
# ============================================================================
def test_7_glm47_exact_dims(device, grid):
    """
    TEST 7: GLM-4.7 exact dimension regression test.

    Uses the exact dimensions from the GLM-4.7 Full (355B) and Flash (47B) models:
      - GLM-4.7-Full-TP4: NKV=2, head_dim=128, block_size=64
      - GLM-4.7-Flash: NKV=2, head_dim=128, block_size=64
      - GLM-4.7-Flash-MLA: NKV=1, head_dim=576 (kv_lora_rank + rope_dim)
    """
    print("\n--- TEST 7: GLM-4.7 exact dimension test ---")
    torch.manual_seed(SEED)

    configs = [
        ("GLM-4.7-Full-TP4", 1, 2, 128, 64, 4),
        ("GLM-4.7-Flash", 1, 2, 128, 64, 4),
        ("GLM-4.7-Flash-MLA", 1, 1, 576, 64, 4),
    ]

    all_passed = True
    for config_name, batch, num_heads, head_dim, block_size, num_blocks in configs:
        test_name = f"7: {config_name} (heads={num_heads}, dim={head_dim})"
        try:
            cache_shape = (batch * num_blocks, num_heads, block_size, head_dim)
            page_table = torch.arange(num_blocks).unsqueeze(0).int()

            tt_cache = ttnn.from_torch(
                torch.zeros(cache_shape).bfloat16(),
                device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            )
            tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

            test_positions = [0, 31, 32, 63, 64, 127]
            position_passed = True

            for pos in test_positions:
                if pos >= block_size * num_blocks:
                    continue

                torch.manual_seed(SEED + pos)
                update_data = torch.randn(1, batch, num_heads, head_dim).bfloat16().float()
                tt_update = prepare_sharded_update(update_data, num_heads, head_dim, batch, device, grid)

                tt_pos = ttnn.from_torch(
                    torch.tensor([pos], dtype=torch.int32), device=device, dtype=ttnn.int32
                )
                ttnn.experimental.paged_update_cache(
                    tt_cache, tt_update,
                    update_idxs_tensor=tt_pos, page_table=tt_page_table,
                )

                result = ttnn.to_torch(ttnn.from_device(tt_cache))
                block_idx = pos // block_size
                row_in_block = pos % block_size
                got = result[block_idx, :num_heads, row_in_block, :]
                want = update_data[0, 0, :num_heads, :].bfloat16()

                if not torch.equal(got, want):
                    err = (got.float() - want.float()).abs().max().item()
                    ok, pcc_val = comp_pcc(want, got, threshold=0.98)
                    record(f"{test_name} pos={pos}", "FAIL",
                           f"max_err={err:.6f}, PCC={pcc_val:.6f}")
                    position_passed = False

                ttnn.deallocate(tt_update)
                ttnn.deallocate(tt_pos)

            if position_passed:
                record(test_name, "PASS", "All positions correct")
            else:
                all_passed = False

            ttnn.deallocate(tt_cache)
            ttnn.deallocate(tt_page_table)

        except Exception as e:
            record(test_name, "ERROR", f"{type(e).__name__}: {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


# ============================================================================
# TEST 8: paged_fill_cache baseline (sanity check -- must ALWAYS pass)
# ============================================================================
def test_8_fill_cache_sanity(device, grid):
    """
    TEST 8: paged_fill_cache sanity check.

    This test MUST pass on both WH and BH. If it fails, the hardware is
    broken or the test environment is misconfigured. paged_fill_cache uses
    direct tile writes (no untilize/tilize), so it bypasses the BH bug.
    """
    print("\n--- TEST 8: paged_fill_cache sanity (must always pass) ---")
    torch.manual_seed(SEED)

    batch = 1
    num_heads = 2
    head_dim = 128
    block_size = 64
    num_blocks = 4
    fill_len = 128

    cache_shape = (batch * num_blocks, num_heads, block_size, head_dim)
    page_table = torch.arange(num_blocks).unsqueeze(0).int()

    try:
        fill_data = torch.randn(1, num_heads, fill_len, head_dim).bfloat16()

        tt_cache = ttnn.from_torch(
            torch.zeros(cache_shape).bfloat16(),
            device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )
        tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)
        tt_fill = ttnn.from_torch(fill_data, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        ttnn.experimental.paged_fill_cache(tt_cache, tt_fill, tt_page_table, batch_idx=0)

        result = ttnn.to_torch(ttnn.from_device(tt_cache))

        result_unpaged = (
            result
            .reshape(batch, num_blocks, num_heads, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, num_heads, block_size * num_blocks, head_dim)
        )

        filled_region = result_unpaged[0, :, :fill_len, :]
        exact_match = torch.equal(filled_region, fill_data[0])

        if exact_match:
            record("8: fill_cache sanity", "PASS", "Exact match")
        else:
            max_err = (filled_region.float() - fill_data[0].float()).abs().max().item()
            record("8: fill_cache sanity", "FAIL",
                   f"paged_fill_cache BROKEN! max_err={max_err:.8f}. Hardware issue?")

        unfilled = result_unpaged[0, :, fill_len:, :]
        zeros_ok = torch.all(unfilled == 0).item()
        if not zeros_ok:
            record("8: fill_cache zeros", "FAIL", "Unfilled region is not zero")
            exact_match = False

        ttnn.deallocate(tt_fill)
        ttnn.deallocate(tt_cache)
        ttnn.deallocate(tt_page_table)

        return exact_match

    except Exception as e:
        record("8: fill_cache sanity", "ERROR", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Pytest wrappers (for CI integration)
# ============================================================================
if HAS_PYTEST:

    @pytest.fixture(scope="module")
    def bh_device():
        """Open a single device for all tests in this module."""
        device = ttnn.CreateDevice(device_id=0)
        yield device
        ttnn.close_device(device)

    @pytest.fixture(scope="module")
    def bh_grid(bh_device):
        return bh_device.compute_with_storage_grid_size()

    @pytest.mark.skipif(
        os.environ.get("TT_ENABLE_HW_TESTS") != "1",
        reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device).",
    )
    class TestPagedUpdateCacheBH:
        """
        paged_update_cache correctness tests targeting the Blackhole untilize/tilize bug.

        BUG: Issue #14594 -- pack_untilize_block/tilize_block pipeline produces wrong
        values on Blackhole hardware. paged_update_cache uses this pipeline for its
        read-modify-write of individual cache rows, causing KV cache corruption.

        Expected behavior:
          - FAIL on current BH until the untilize/tilize kernel fix is applied
          - PASS on Wormhole (regression protection)

        Blackhole grid: 13x10 = 130 cores (12x10 = 120 usable after harvest)
        Wormhole grid: 8x8 = 64 cores (T3K) or 8x9 = 72 cores (Galaxy)
        """

        def test_fill_cache_sanity(self, bh_device, bh_grid):
            """Sanity: paged_fill_cache must work (bypasses untilize/tilize)."""
            assert test_8_fill_cache_sanity(bh_device, bh_grid), "paged_fill_cache is broken -- hardware issue"

        def test_single_row_update(self, bh_device, bh_grid):
            """Core bug: single-row update via pack_untilize_block/tilize_block."""
            assert test_1_single_row_update(bh_device, bh_grid)

        def test_tile_boundary_stress(self, bh_device, bh_grid):
            """Stress: every position 0..63 in the untilize/tilize pipeline."""
            assert test_2_tile_boundary_stress(bh_device, bh_grid)

        def test_multi_step_decode(self, bh_device, bh_grid):
            """Regression: sequential updates must not corrupt earlier writes."""
            assert test_3_multi_step_decode(bh_device, bh_grid)

        def test_sdpa_integration(self, bh_device, bh_grid):
            """Integration: cache update + SDPA decode end-to-end."""
            assert test_4_cache_update_sdpa_integration(bh_device, bh_grid)

        def test_multi_user(self, bh_device, bh_grid):
            """Multi-user: no cross-user corruption with shuffled page tables."""
            assert test_5_multi_user_update(bh_device, bh_grid)

        def test_canary_values(self, bh_device, bh_grid):
            """Canary: constant values make corruption patterns visible."""
            assert test_6_canary_values(bh_device, bh_grid)

        def test_glm47_exact_dims(self, bh_device, bh_grid):
            """Model regression: exact GLM-4.7 cache dimensions."""
            assert test_7_glm47_exact_dims(bh_device, bh_grid)


# ============================================================================
# Standalone runner
# ============================================================================
def main():
    print("=" * 70)
    print("paged_update_cache Blackhole Correctness Tests")
    print("Issue #14594: pack_untilize_block/tilize_block broken on BH")
    print("=" * 70)

    device = ttnn.CreateDevice(device_id=0)
    grid = device.compute_with_storage_grid_size()
    arch = device.arch()
    print(f"Device arch: {arch}")
    print(f"Compute grid: {grid.x}x{grid.y} = {grid.x * grid.y} cores")
    print(f"Expected: BH=13x10(130), WH-T3K=8x8(64), WH-Galaxy=8x9(72)")
    print("=" * 70)

    # Run tests in order of increasing complexity
    # Test 8 (fill_cache) MUST pass -- it's the sanity baseline
    test_8_fill_cache_sanity(device, grid)
    test_6_canary_values(device, grid)
    test_1_single_row_update(device, grid)
    test_2_tile_boundary_stress(device, grid)
    test_3_multi_step_decode(device, grid)
    test_5_multi_user_update(device, grid)
    test_4_cache_update_sdpa_integration(device, grid)
    test_7_glm47_exact_dims(device, grid)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    total = len(ALL_RESULTS)
    passes = sum(1 for r in ALL_RESULTS if r.status == "PASS")
    fails = sum(1 for r in ALL_RESULTS if r.status == "FAIL")
    errors = sum(1 for r in ALL_RESULTS if r.status == "ERROR")

    print(f"  PASSED:  {passes}/{total}")
    print(f"  FAILED:  {fails}/{total}")
    print(f"  ERRORS:  {errors}/{total}")

    if fails > 0:
        print("\n  FAILED TESTS:")
        for r in ALL_RESULTS:
            if r.status == "FAIL":
                print(f"    - {r.name}: {r.detail}")

    if errors > 0:
        print("\n  ERROR TESTS:")
        for r in ALL_RESULTS:
            if r.status == "ERROR":
                print(f"    - {r.name}: {r.detail}")

    print("=" * 70)
    if fails == 0 and errors == 0:
        print("ALL TESTS PASSED")
        print("If running on BH, this means the untilize/tilize fix is working!")
    else:
        print("TESTS FAILED")
        print("If running on BH, this confirms Issue #14594 (pack_untilize_block/tilize_block bug)")
        print("If running on WH, this is a REGRESSION -- investigate immediately!")
    print("=" * 70)

    ttnn.close_device(device)
    sys.exit(0 if (fails == 0 and errors == 0) else 1)


if __name__ == "__main__":
    main()

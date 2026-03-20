"""Incremental SDPA decode tests for Blackhole Galaxy — find where garbling starts."""
import torch
import ttnn
import numpy as np
import sys

torch.manual_seed(42)
np.random.seed(42)

# GLM-4.7 dimensions at TP=4
B, NH, NKV, D = 1, 24, 2, 128  # batch=1, 24 Q heads, 2 KV heads, dim=128
BLOCK_SIZE = 64
NUM_BLOCKS = 4  # 4 blocks × 64 = 256 token capacity
PREFIX_LEN = 5  # short prefix for testing
SCALE = D ** -0.5
GQA_RATIO = NH // NKV


def torch_sdpa_ref(q, k, v, cur_pos, scale):
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
        mask[0, :pos + 1] = 0.0
        for kvi in range(nkv):
            qi = q[0, bi, kvi * gqa:(kvi + 1) * gqa, :]
            ki = k[bi, kvi, :, :]
            vi = v[bi, kvi, :, :]
            sc = torch.matmul(qi, ki.T) * scale + mask
            w = torch.softmax(sc.float(), dim=-1).to(vi.dtype)
            outputs.append(torch.matmul(w, vi))
    result = torch.stack(outputs, dim=0).reshape(b, nh, d)
    return result.unsqueeze(0)


def comp_pcc(golden, calculated, threshold=0.99):
    g = golden.float().flatten()
    c = calculated.float().flatten()
    if g.shape != c.shape:
        return False, 0.0
    if torch.all(g == 0) and torch.all(c == 0):
        return True, 1.0
    pcc = torch.corrcoef(torch.stack([g, c]))[0, 1].item()
    if np.isnan(pcc):
        return False, 0.0
    return pcc >= threshold, pcc


print("=" * 60)
print("INCREMENTAL SDPA DECODE TESTS — Blackhole Galaxy")
print("=" * 60)

device = ttnn.CreateDevice(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: {device.arch()}, Grid: {grid.x}x{grid.y}")

sdpa_cfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(min(8, grid.x), min(8, grid.y)),
    q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False,
)
cc_cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=False,
)

results = []

# ============================================================
# TEST 1: Paged SDPA decode (read-only, host-built cache)
# ============================================================
print("\n--- TEST 1: Paged SDPA decode (read-only) ---")
try:
    # Build KV cache on host, fill with known data
    seq_len = BLOCK_SIZE * NUM_BLOCKS  # 256
    k_host = torch.randn(B, NKV, seq_len, D).float()
    v_host = torch.randn(B, NKV, seq_len, D).float()
    q_host = torch.randn(1, B, NH, D).float()
    cur_pos = [PREFIX_LEN - 1]  # last position of prefix

    # Create paged KV cache tensors
    tt_k_cache = ttnn.from_torch(
        k_host.reshape(B, NKV, NUM_BLOCKS, BLOCK_SIZE, D).permute(0, 2, 1, 3, 4).reshape(B * NUM_BLOCKS, NKV, BLOCK_SIZE, D),
        device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
    )
    tt_v_cache = ttnn.from_torch(
        v_host.reshape(B, NKV, NUM_BLOCKS, BLOCK_SIZE, D).permute(0, 2, 1, 3, 4).reshape(B * NUM_BLOCKS, NKV, BLOCK_SIZE, D),
        device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
    )

    # Page table: identity mapping (block i → page i)
    page_table = torch.arange(NUM_BLOCKS).unsqueeze(0).int()  # [1, NUM_BLOCKS]
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

    tt_q = ttnn.from_torch(q_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_cur_pos = ttnn.from_torch(torch.tensor(cur_pos), device=device, dtype=ttnn.int32)

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q, tt_k_cache, tt_v_cache,
        page_table_tensor=tt_page_table,
        cur_pos_tensor=tt_cur_pos,
        scale=SCALE,
        program_config=sdpa_cfg,
        compute_kernel_config=cc_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.to_torch(tt_out)[..., :NH, :D]
    ref = torch_sdpa_ref(q_host, k_host, v_host, cur_pos, SCALE)
    passed, pcc = comp_pcc(ref, tt_result, 0.99)
    max_err = (ref - tt_result).abs().max().item()
    print(f"  PCC: {pcc:.6f}, Max err: {max_err:.6f} -> {'PASS' if passed else 'FAIL'}")
    results.append(("1: Paged SDPA read-only", "PASS" if passed else "FAIL", pcc))

    ttnn.deallocate(tt_q); ttnn.deallocate(tt_k_cache); ttnn.deallocate(tt_v_cache)
    ttnn.deallocate(tt_out)
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("1: Paged SDPA read-only", f"ERROR", 0.0))

# ============================================================
# TEST 2: paged_fill_cache + paged SDPA decode (fill→read cycle)
# ============================================================
print("\n--- TEST 2: paged_fill_cache + paged SDPA decode ---")
try:
    k_prefix = torch.randn(1, NKV, PREFIX_LEN, D).float()
    v_prefix = torch.randn(1, NKV, PREFIX_LEN, D).float()
    q_host = torch.randn(1, B, NH, D).float()

    # Allocate empty cache
    cache_shape = (B * NUM_BLOCKS, NKV, BLOCK_SIZE, D)
    tt_k_cache = ttnn.from_torch(torch.zeros(cache_shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_cache = ttnn.from_torch(torch.zeros(cache_shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    page_table = torch.arange(NUM_BLOCKS).unsqueeze(0).int()
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

    # Fill cache with prefix
    tt_k_prefix = ttnn.from_torch(k_prefix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_prefix = ttnn.from_torch(v_prefix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn.experimental.paged_fill_cache(tt_k_cache, tt_k_prefix, tt_page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(tt_v_cache, tt_v_prefix, tt_page_table, batch_idx=0)

    ttnn.deallocate(tt_k_prefix); ttnn.deallocate(tt_v_prefix)

    # Now do SDPA decode reading from the filled cache
    tt_q = ttnn.from_torch(q_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cur_pos = [PREFIX_LEN - 1]
    tt_cur_pos = ttnn.from_torch(torch.tensor(cur_pos), device=device, dtype=ttnn.int32)

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q, tt_k_cache, tt_v_cache,
        page_table_tensor=tt_page_table,
        cur_pos_tensor=tt_cur_pos,
        scale=SCALE,
        program_config=sdpa_cfg,
        compute_kernel_config=cc_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.to_torch(tt_out)[..., :NH, :D]

    # Build torch reference from prefix (zero-padded to full cache size)
    k_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
    v_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
    k_full[0, :, :PREFIX_LEN, :] = k_prefix[0]
    v_full[0, :, :PREFIX_LEN, :] = v_prefix[0]
    ref = torch_sdpa_ref(q_host, k_full, v_full, cur_pos, SCALE)

    passed, pcc = comp_pcc(ref, tt_result, 0.98)
    max_err = (ref - tt_result).abs().max().item()
    print(f"  PCC: {pcc:.6f}, Max err: {max_err:.6f} -> {'PASS' if passed else 'FAIL'}")
    results.append(("2: Fill + SDPA read", "PASS" if passed else "FAIL", pcc))

    ttnn.deallocate(tt_q); ttnn.deallocate(tt_k_cache); ttnn.deallocate(tt_v_cache)
    ttnn.deallocate(tt_out)
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("2: Fill + SDPA read", f"ERROR", 0.0))

# ============================================================
# TEST 3: Fill→Update→Read cycle (simulates prefill + 1 decode step)
# ============================================================
print("\n--- TEST 3: Fill + Update + SDPA decode (full cycle) ---")
try:
    k_prefix = torch.randn(1, NKV, PREFIX_LEN, D).float()
    v_prefix = torch.randn(1, NKV, PREFIX_LEN, D).float()
    k_new = torch.randn(1, B, NKV, D).float()  # new token K
    v_new = torch.randn(1, B, NKV, D).float()  # new token V
    q_decode = torch.randn(1, B, NH, D).float()  # decode query

    # Allocate cache
    cache_shape = (B * NUM_BLOCKS, NKV, BLOCK_SIZE, D)
    tt_k_cache = ttnn.from_torch(torch.zeros(cache_shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_cache = ttnn.from_torch(torch.zeros(cache_shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    page_table = torch.arange(NUM_BLOCKS).unsqueeze(0).int()
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

    # Step 1: Fill cache with prefix
    tt_k_prefix = ttnn.from_torch(k_prefix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_prefix = ttnn.from_torch(v_prefix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.experimental.paged_fill_cache(tt_k_cache, tt_k_prefix, tt_page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(tt_v_cache, tt_v_prefix, tt_page_table, batch_idx=0)
    ttnn.deallocate(tt_k_prefix); ttnn.deallocate(tt_v_prefix)

    # Step 2: Update cache with new decode token at position PREFIX_LEN
    update_pos = [PREFIX_LEN]
    tt_update_pos = ttnn.from_torch(torch.tensor(update_pos), device=device, dtype=ttnn.int32)

    # paged_update_cache expects HEIGHT_SHARDED input
    tt_k_new = ttnn.from_torch(k_new, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_new = ttnn.from_torch(v_new, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # Shard K/V for paged_update_cache
    kv_shard_grid = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
    kv_shard_spec = ttnn.ShardSpec(kv_shard_grid, [NKV * 32, D], ttnn.ShardOrientation.ROW_MAJOR)
    kv_shard_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_shard_spec)
    tt_k_new = ttnn.interleaved_to_sharded(tt_k_new, kv_shard_mc)
    tt_v_new = ttnn.interleaved_to_sharded(tt_v_new, kv_shard_mc)

    ttnn.experimental.paged_update_cache(tt_k_cache, tt_k_new, update_idxs_tensor=tt_update_pos, page_table=tt_page_table)
    ttnn.experimental.paged_update_cache(tt_v_cache, tt_v_new, update_idxs_tensor=tt_update_pos, page_table=tt_page_table)
    ttnn.deallocate(tt_k_new); ttnn.deallocate(tt_v_new)

    # Step 3: SDPA decode at position PREFIX_LEN (reads prefix + new token)
    cur_pos = [PREFIX_LEN]
    tt_cur_pos = ttnn.from_torch(torch.tensor(cur_pos), device=device, dtype=ttnn.int32)
    tt_q = ttnn.from_torch(q_decode, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q, tt_k_cache, tt_v_cache,
        page_table_tensor=tt_page_table,
        cur_pos_tensor=tt_cur_pos,
        scale=SCALE,
        program_config=sdpa_cfg,
        compute_kernel_config=cc_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.to_torch(tt_out)[..., :NH, :D]

    # Build torch reference
    k_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
    v_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
    k_full[0, :, :PREFIX_LEN, :] = k_prefix[0]
    v_full[0, :, :PREFIX_LEN, :] = v_prefix[0]
    k_full[0, :, PREFIX_LEN, :] = k_new[0, 0, :, :]
    v_full[0, :, PREFIX_LEN, :] = v_new[0, 0, :, :]
    ref = torch_sdpa_ref(q_decode, k_full, v_full, cur_pos, SCALE)

    passed, pcc = comp_pcc(ref, tt_result, 0.98)
    max_err = (ref - tt_result).abs().max().item()
    print(f"  PCC: {pcc:.6f}, Max err: {max_err:.6f} -> {'PASS' if passed else 'FAIL'}")
    results.append(("3: Fill+Update+Read cycle", "PASS" if passed else "FAIL", pcc))

    ttnn.deallocate(tt_q); ttnn.deallocate(tt_k_cache); ttnn.deallocate(tt_v_cache)
    ttnn.deallocate(tt_out)
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("3: Fill+Update+Read cycle", f"ERROR", 0.0))

# ============================================================
# TEST 4: Multiple decode steps (simulates token-by-token generation)
# ============================================================
print("\n--- TEST 4: Multiple decode steps (5 tokens after prefix) ---")
try:
    k_prefix = torch.randn(1, NKV, PREFIX_LEN, D).float()
    v_prefix = torch.randn(1, NKV, PREFIX_LEN, D).float()

    cache_shape = (B * NUM_BLOCKS, NKV, BLOCK_SIZE, D)
    tt_k_cache = ttnn.from_torch(torch.zeros(cache_shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_cache = ttnn.from_torch(torch.zeros(cache_shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    page_table = torch.arange(NUM_BLOCKS).unsqueeze(0).int()
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)

    # Fill prefix
    tt_k_prefix = ttnn.from_torch(k_prefix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v_prefix = ttnn.from_torch(v_prefix, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.experimental.paged_fill_cache(tt_k_cache, tt_k_prefix, tt_page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(tt_v_cache, tt_v_prefix, tt_page_table, batch_idx=0)
    ttnn.deallocate(tt_k_prefix); ttnn.deallocate(tt_v_prefix)

    # Track full K/V for reference
    k_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
    v_full = torch.zeros(B, NKV, BLOCK_SIZE * NUM_BLOCKS, D)
    k_full[0, :, :PREFIX_LEN, :] = k_prefix[0]
    v_full[0, :, :PREFIX_LEN, :] = v_prefix[0]

    all_passed = True
    for step in range(5):
        pos = PREFIX_LEN + step
        k_new = torch.randn(1, B, NKV, D).float()
        v_new = torch.randn(1, B, NKV, D).float()
        q_decode = torch.randn(1, B, NH, D).float()

        # Update cache (paged_update_cache requires HEIGHT_SHARDED input)
        tt_pos = ttnn.from_torch(torch.tensor([pos]), device=device, dtype=ttnn.int32)
        tt_k_new = ttnn.from_torch(k_new, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tt_v_new = ttnn.from_torch(v_new, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        kv_shard_grid = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
        kv_shard_spec = ttnn.ShardSpec(kv_shard_grid, [NKV * 32, D], ttnn.ShardOrientation.ROW_MAJOR, False)
        kv_shard_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_shard_spec)
        tt_k_new = ttnn.interleaved_to_sharded(tt_k_new, kv_shard_mc)
        tt_v_new = ttnn.interleaved_to_sharded(tt_v_new, kv_shard_mc)
        ttnn.experimental.paged_update_cache(tt_k_cache, tt_k_new, update_idxs_tensor=tt_pos, page_table=tt_page_table)
        ttnn.experimental.paged_update_cache(tt_v_cache, tt_v_new, update_idxs_tensor=tt_pos, page_table=tt_page_table)
        ttnn.deallocate(tt_k_new); ttnn.deallocate(tt_v_new)

        # Update reference
        k_full[0, :, pos, :] = k_new[0, 0, :, :]
        v_full[0, :, pos, :] = v_new[0, 0, :, :]

        # SDPA decode
        tt_q = ttnn.from_torch(q_decode, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
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
        passed, pcc = comp_pcc(ref, tt_result, 0.98)
        max_err = (ref - tt_result).abs().max().item()
        status = "PASS" if passed else "FAIL"
        print(f"  Step {step} (pos={pos}): PCC={pcc:.6f}, err={max_err:.6f} -> {status}")
        if not passed:
            all_passed = False

        ttnn.deallocate(tt_q); ttnn.deallocate(tt_out)

    results.append(("4: Multi-step decode (5 steps)", "PASS" if all_passed else "FAIL", pcc))

    ttnn.deallocate(tt_k_cache); ttnn.deallocate(tt_v_cache)
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("4: Multi-step decode", f"ERROR", 0.0))

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
all_pass = True
for name, status, pcc in results:
    flag = "OK" if status == "PASS" else "XX"
    print(f"  [{flag}] {status} (PCC={pcc:.6f}): {name}")
    if status != "PASS":
        all_pass = False
print("=" * 60)
print("ALL PASSED" if all_pass else "SOME FAILED — see above")
print("=" * 60)

ttnn.close_device(device)

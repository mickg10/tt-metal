"""SDPA decode correctness test for Blackhole Galaxy — with torch reference PCC."""
import torch
import ttnn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


def torch_sdpa_decode_reference(q, k, v, cur_pos, scale):
    """Torch reference for SDPA decode (last-token attention).

    Args:
        q: [1, B, NH, D] — query for current decode step
        k: [B, NKV, S, D] — full key cache
        v: [B, NKV, S, D] — full value cache
        cur_pos: list of int — current position per batch (for causal mask)
        scale: float — attention scale (1/sqrt(d))

    Returns:
        [1, B, NH, D] — attention output
    """
    B = q.shape[1]
    NH = q.shape[2]
    NKV = k.shape[1]
    S = k.shape[2]
    D = q.shape[3]
    GQA_RATIO = NH // NKV

    outputs = []
    for b in range(B):
        pos = cur_pos[b]
        # Build causal mask: only attend to positions 0..pos
        mask = torch.full((1, S), float("-inf"))
        mask[0, : pos + 1] = 0.0

        for kv_head in range(NKV):
            q_heads = q[0, b, kv_head * GQA_RATIO : (kv_head + 1) * GQA_RATIO, :]  # [GQA, D]
            k_head = k[b, kv_head, :, :]  # [S, D]
            v_head = v[b, kv_head, :, :]  # [S, D]

            # QK^T: [GQA, D] x [D, S] = [GQA, S]
            scores = torch.matmul(q_heads, k_head.transpose(-2, -1)) * scale
            scores = scores + mask  # apply causal mask
            attn_weights = torch.softmax(scores.float(), dim=-1).to(v_head.dtype)
            # Attn x V: [GQA, S] x [S, D] = [GQA, D]
            out = torch.matmul(attn_weights, v_head)
            outputs.append(out)

    # Reassemble: [B * NKV * GQA, D] -> [1, B, NH, D]
    result = torch.stack(outputs, dim=0)  # [B*NKV, GQA, D]
    result = result.reshape(B, NKV, GQA_RATIO, D)
    result = result.reshape(B, NH, D)
    return result.unsqueeze(0)  # [1, B, NH, D]


def comp_pcc(golden, calculated, pcc_threshold=0.99):
    """Compute Pearson Correlation Coefficient between golden and calculated tensors."""
    golden = golden.float().flatten()
    calculated = calculated.float().flatten()

    if golden.shape != calculated.shape:
        return False, 0.0

    if torch.all(golden == 0) and torch.all(calculated == 0):
        return True, 1.0

    pcc = torch.corrcoef(torch.stack([golden, calculated]))[0, 1].item()
    if np.isnan(pcc):
        return False, 0.0
    return pcc >= pcc_threshold, pcc


configs = [
    # (batch, num_q_heads, num_kv_heads, seq_len, head_dim, name, pcc_threshold)
    (4, 8, 1, 1024, 128, "Llama70B-like (nkv=1, nh=8)", 0.99),
    (4, 32, 8, 1024, 128, "GLM-4.7 TP=4 (nkv=8, nh=32)", 0.99),
    (4, 12, 1, 1024, 128, "GLM-4.7 TP=8 (nkv=1, nh=12)", 0.99),
    (1, 32, 4, 256, 64, "GPT-OSS sink-like (nkv=4, d=64)", 0.99),
    (32, 32, 1, 128, 128, "batch=32 nkv=1", 0.99),
    (32, 8, 8, 128, 128, "batch=32 nkv=8 (GQA ratio=1)", 0.99),
    # Smaller seq for faster execution
    (4, 24, 2, 512, 128, "GLM-4.7 TP=4 exact (24Q/2KV)", 0.99),
    (1, 96, 8, 256, 128, "GLM-4.7 full heads (96Q/8KV)", 0.99),
]

# When running inside a vLLM container, devices are already open.
# Use ttnn.GetDefaultDevice() to get the existing device.
# When running standalone, open a device.
device = ttnn.GetDefaultDevice()
if device is None:
    device = ttnn.CreateDevice(device_id=0)
    ttnn.SetDefaultDevice(device)
    standalone = True
    print("Opened device standalone")
else:
    standalone = False
    print("Using existing device from vLLM container")
grid = device.compute_with_storage_grid_size()
print(f"BH compute grid: {grid.x}x{grid.y}")
print(f"Architecture: {device.arch()}")
print()

results = []
for b, nh, nkv, s, d, name, pcc_thresh in configs:
    print(f"Test: {name} (b={b}, nh={nh}, nkv={nkv}, s={s}, d={d})")

    # Generate random data (same for torch and ttnn)
    q_torch = torch.randn(1, b, nh, d).float()
    k_torch = torch.randn(b, nkv, s, d).float()
    v_torch = torch.randn(b, nkv, s, d).float()
    cur_pos = [s - 1] * b  # attend to full sequence
    scale = d ** -0.5

    # Torch reference
    ref_out = torch_sdpa_decode_reference(q_torch, k_torch, v_torch, cur_pos, scale)

    # TTNN execution
    try:
        tt_q = ttnn.from_torch(q_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tt_k = ttnn.from_torch(k_torch, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        tt_v = ttnn.from_torch(v_torch, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        tt_pos = ttnn.from_torch(torch.tensor(cur_pos), device=device, dtype=ttnn.int32)

        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(min(8, grid.x), min(8, grid.y)),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q, tt_k, tt_v,
            cur_pos_tensor=tt_pos,
            scale=scale,
            program_config=cfg,
            compute_kernel_config=cc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_out_torch = ttnn.to_torch(tt_out)

        # Trim padding if needed (TTNN may pad heads to tile boundary)
        tt_out_trimmed = tt_out_torch[..., :nh, :d]

        # Compute PCC
        passed, pcc = comp_pcc(ref_out, tt_out_trimmed, pcc_thresh)

        # Also compute max absolute error
        max_abs_err = (ref_out - tt_out_trimmed).abs().max().item()
        mean_abs_err = (ref_out - tt_out_trimmed).abs().mean().item()

        status = "PASS" if passed else "FAIL"
        print(f"  PCC: {pcc:.6f} (threshold: {pcc_thresh})")
        print(f"  Max abs error: {max_abs_err:.6f}, Mean abs error: {mean_abs_err:.6f}")
        print(f"  Ref  range: [{ref_out.min():.4f}, {ref_out.max():.4f}], mean={ref_out.abs().mean():.4f}")
        print(f"  TTNN range: [{tt_out_trimmed.min():.4f}, {tt_out_trimmed.max():.4f}], mean={tt_out_trimmed.abs().mean():.4f}")
        print(f"  -> {status}")
        results.append((name, status, pcc))

        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)
        ttnn.deallocate(tt_out)

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append((name, "ERROR", 0.0))

if standalone:
    ttnn.close_device(device)
else:
    print("Skipping device close (managed by vLLM)")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
all_pass = True
for name, status, pcc in results:
    flag = "✓" if status == "PASS" else "✗"
    print(f"  {flag} {status} (PCC={pcc:.6f}): {name}")
    if status != "PASS":
        all_pass = False
print("=" * 60)
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 60)

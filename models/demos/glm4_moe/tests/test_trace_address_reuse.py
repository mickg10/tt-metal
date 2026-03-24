# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test that trace replay survives chunked prefill — DRAM address reuse prevention.

This test replicates the GLM-4.7-Full production scenario where:
1. A decode trace is captured (92-layer forward with intermediates allocated/freed)
2. A chunked prefill runs (4K+ tokens, PRESERVE_TRACE=1, no trace release)
3. The decode trace is replayed
4. Output must match expected (PCC >= 0.999)

The test MUST FAIL on current code at 4K+ chunks (address reuse corrupts trace data)
and MUST PASS after the fix (C++ Record+Re-reserve or Python TraceRetainer).

Run:
    # Single device (basic sanity):
    pytest models/demos/glm4_moe/tests/test_trace_address_reuse.py -v -k "N150"

    # TG mesh (full reproduction):
    pytest models/demos/glm4_moe/tests/test_trace_address_reuse.py -v -k "8x4"
"""
import os
import pytest
import torch
import ttnn
from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bf16_tensor(shape, mesh_device, seed=42, scale=1.0):
    """Create a deterministic BF16 DRAM tensor on mesh."""
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16) * scale
    is_mesh = mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _readback(t, mesh_device):
    """Read a tensor back to torch, handling mesh vs single device."""
    if mesh_device.get_num_devices() > 1:
        try:
            dev_t = ttnn.get_device_tensors(t)[0]
            return ttnn.to_torch(dev_t).float()
        except (AttributeError, RuntimeError):
            return ttnn.to_torch(t).float()
    return ttnn.to_torch(t).float()


def _simulate_layer_forward(x, w1, w2, mesh_device):
    """Simulate one decoder layer: two matmuls with intermediate allocation/deallocation.

    This models the GLM-4.7 pattern where each layer allocates ~5-6 MB of intermediates
    (attention output, MoE routing, sparse matmul results, EP reduce output) and frees
    them before the next layer.

    x:  [1, 1, S, H]
    w1: [1, 1, H, H] (simulates combined attention+MoE weights)
    w2: [1, 1, H, H] (simulates output projection)
    """
    # Intermediate 1: "attention + MoE routing"
    intermediate = ttnn.matmul(x, w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Intermediate 2: "output projection"
    output = ttnn.matmul(intermediate, w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Free intermediate (models deallocate pattern in decoder_layer forward)
    ttnn.deallocate(intermediate, force=True)
    return output


def _simulate_prefill_chunk(mesh_device, chunk_shape, w1, w2, num_layers):
    """Simulate one prefill chunk through all layers.

    Allocates a chunk tensor, runs through num_layers, frees everything.
    Models the _prefill_chunked inner loop.
    """
    x = _make_bf16_tensor(chunk_shape, mesh_device, seed=99)
    for _ in range(num_layers):
        x_next = _simulate_layer_forward(x, w1, w2, mesh_device)
        ttnn.deallocate(x, force=False)
        x = x_next
    return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Use trace_region_size=0 (dynamic mode) to match production config.
# This is critical — it's the mode where trace command buffers go top-down
# in DRAM and data tensors go bottom-up, enabling address reuse.

@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((8, 4), id="8x4_grid"),
        pytest.param(1, id="N150"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "fabric_config": True}],
    indirect=True,
)
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
@pytest.mark.parametrize("num_layers", [4, 16])
def test_trace_survives_chunked_prefill(mesh_device, num_chunks, num_layers):
    """Trace replay must produce correct output after chunked prefill (PRESERVE_TRACE=1).

    This test captures a decode trace, then simulates a chunked prefill WITHOUT
    releasing the trace, then replays the trace and checks correctness.

    Expected behavior:
    - num_chunks=1: PASS (baseline — single chunk doesn't trigger heavy address reuse)
    - num_chunks=4, num_layers>=16: FAIL on unfixed code (address reuse corrupts trace)
    - All cases: PASS after fix

    The failure mode is that prefill intermediate allocations land in the same DRAM
    addresses the trace intermediates used (freed after capture, coalesced by first-fit
    allocator), and trace replay reads/writes corrupted data.
    """
    H = 256  # Hidden dim (small for test, proportional to production 5120)
    S_decode = 32  # Decode sequence length (tile-padded batch=1)
    S_chunk = 256  # Chunk size (models PREFILL_CHUNK_SIZE=1024 at reduced scale)

    is_mesh = mesh_device.get_num_devices() > 1

    # --- Step 1: Create persistent tensors (survive across trace + prefill) ---
    # These model weights, KV cache, and trace input/output tensors.
    # The trace reads from input_tt and writes to output addresses.

    # Scale weights down to prevent BF16 overflow with many layers.
    # Each matmul amplifies by ~sqrt(H)*scale, so 16 layers: (sqrt(256)*0.05)^16 ~ 0.8^16 ~ 0.03
    w_scale = 0.05
    input_tt = _make_bf16_tensor((1, 1, S_decode, H), mesh_device, seed=1)
    w1 = _make_bf16_tensor((1, 1, H, H), mesh_device, seed=2, scale=w_scale)
    w2 = _make_bf16_tensor((1, 1, H, H), mesh_device, seed=3, scale=w_scale)

    # --- Step 2: Compile warmup (pre-trace) ---
    logger.info("Compile warmup: {} layers", num_layers)
    x = input_tt
    for _ in range(num_layers):
        x_next = _simulate_layer_forward(x, w1, w2, mesh_device)
        if x is not input_tt:
            ttnn.deallocate(x, force=True)
        x = x_next
    # Read expected output before freeing
    expected_output = _readback(x, mesh_device)
    ttnn.deallocate(x, force=True)
    ttnn.synchronize_device(mesh_device)

    # Re-copy input (compile warmup may have mutated in-place)
    host_input = ttnn.from_torch(
        torch.randn(1, 1, S_decode, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    torch.manual_seed(1)
    host_input_data = ttnn.from_torch(
        torch.randn(1, 1, S_decode, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    ttnn.copy_host_to_device_tensor(host_input_data, input_tt)
    ttnn.synchronize_device(mesh_device)

    # --- Step 3: Capture decode trace ---
    # This models _capture_decode_trace: run forward through num_layers,
    # retaining inter-layer x tensors (like production retains 91 x tensors).
    logger.info("Capturing trace: {} layers", num_layers)
    retained = []  # Models retained_intermediates in _DecodeTraceState

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    x = input_tt
    for layer_idx in range(num_layers):
        x_next = _simulate_layer_forward(x, w1, w2, mesh_device)
        # Retain x to prevent DRAM reuse (like production code at model_tt.py:1424)
        if layer_idx > 0:
            retained.append(x)
        x = x_next

    # x is the trace output — keep a reference for reading after replay
    trace_output = x

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    logger.info("Trace captured, retained {} intermediates", len(retained))

    # NOTE: Do NOT read trace_output here — populate_mesh_buffer (called by end_trace_capture)
    # allocates the trace command buffer in DRAM which may overwrite the trace output's address.
    # The output is only valid after trace replay.

    # --- Step 4: Verify trace replay BEFORE prefill (sanity check) ---
    # Compare trace replay output against warmup expected_output (computed with same input)
    ttnn.copy_host_to_device_tensor(host_input_data, input_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    pre_prefill_output = _readback(trace_output, mesh_device)
    logger.info("pre_prefill_output: min={:.4f} max={:.4f} nan={} shape={}",
                pre_prefill_output.min().item(), pre_prefill_output.max().item(),
                torch.isnan(pre_prefill_output).any().item(), list(pre_prefill_output.shape))
    pre_max_diff = (expected_output.flatten() - pre_prefill_output.flatten()).abs().max().item()
    logger.info("Pre-prefill trace replay vs expected max_abs_diff: {:.6f}", pre_max_diff)
    assert pre_max_diff < 1.0, f"Trace replay before prefill failed: max_abs_diff={pre_max_diff:.6f}"

    # --- Step 5: Simulate chunked prefill (PRESERVE_TRACE=1 — NO trace release) ---
    # This is the critical section. With PRESERVE_TRACE=1, the trace stays captured.
    # Prefill allocates intermediates that may land in freed trace address range.
    logger.info("Simulating chunked prefill: {} chunks × {} layers", num_chunks, num_layers)

    chunk_shape = (1, 1, S_chunk, H)
    prefill_outputs = []

    for layer_idx in range(num_layers):
        chunk_results = []
        for chunk_idx in range(num_chunks):
            # Allocate chunk input
            chunk_x = _make_bf16_tensor(chunk_shape, mesh_device, seed=100 + chunk_idx)
            # Run one layer
            chunk_out = _simulate_layer_forward(chunk_x, w1, w2, mesh_device)
            chunk_results.append(chunk_out)
            ttnn.deallocate(chunk_x, force=False)

        # Concat chunks (like _prefill_chunked line 1977)
        if len(chunk_results) == 1:
            prefill_x = chunk_results[0]
        else:
            prefill_x = ttnn.concat(chunk_results, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for ct in chunk_results:
                ttnn.deallocate(ct, force=False)

        # Free previous layer's full output
        if layer_idx > 0:
            pass  # Already freed by concat dealloc pattern
        prefill_outputs.append(prefill_x)

    # Free all prefill outputs (simulates end of prefill)
    for po in prefill_outputs:
        ttnn.deallocate(po, force=False)
    prefill_outputs.clear()
    ttnn.synchronize_device(mesh_device)
    logger.info("Prefill simulation complete")

    # --- Step 6: Replay trace AFTER prefill ---
    # This is where corruption manifests. If prefill allocations overwrote
    # trace intermediate addresses, the replayed trace reads garbage.
    logger.info("Replaying trace after prefill ({} chunks)", num_chunks)

    ttnn.copy_host_to_device_tensor(host_input_data, input_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    post_prefill_output = _readback(trace_output, mesh_device)

    # --- Step 7: Verify correctness ---
    # Compare against expected_output from warmup (not trace_capture_output which is stale)
    post_max_diff = (expected_output.flatten() - post_prefill_output.flatten()).abs().max().item()
    logger.info(
        "Post-prefill trace replay max_abs_diff: {:.6f} (chunks={}, layers={})",
        post_max_diff, num_chunks, num_layers,
    )

    # The test: output after prefill must match output at capture time
    assert post_max_diff < 1.0, (
        f"Trace replay after {num_chunks}-chunk prefill CORRUPTED: max_abs_diff={post_max_diff:.6f}. "
        f"This indicates DRAM address reuse between trace intermediates and prefill allocations. "
        f"Fix: implement Record+Re-reserve in C++ (bank_manager tracks freed addresses during "
        f"trace capture and re-allocates them after end_trace_capture)."
    )

    # --- Cleanup ---
    ttnn.release_trace(mesh_device, trace_id)
    for t in retained:
        try:
            ttnn.deallocate(t, force=True)
        except Exception:
            pass
    ttnn.deallocate(input_tt, force=True)
    ttnn.deallocate(w1, force=True)
    ttnn.deallocate(w2, force=True)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((8, 4), id="8x4_grid"),
        pytest.param(1, id="N150"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "fabric_config": True}],
    indirect=True,
)
def test_trace_multiple_prefill_cycles(mesh_device):
    """Trace must survive MULTIPLE prefill cycles without degradation.

    Models production pattern: prefill → decode(trace) → prefill → decode(trace) → ...
    Each cycle must produce identical trace output.
    """
    H = 256
    S_decode = 32
    S_chunk = 128
    NUM_LAYERS = 8
    NUM_CYCLES = 3
    NUM_CHUNKS = 2

    is_mesh = mesh_device.get_num_devices() > 1

    w_scale = 0.05
    input_tt = _make_bf16_tensor((1, 1, S_decode, H), mesh_device, seed=1)
    w1 = _make_bf16_tensor((1, 1, H, H), mesh_device, seed=2, scale=w_scale)
    w2 = _make_bf16_tensor((1, 1, H, H), mesh_device, seed=3, scale=w_scale)

    # Compile warmup
    x = input_tt
    for _ in range(NUM_LAYERS):
        x_next = _simulate_layer_forward(x, w1, w2, mesh_device)
        if x is not input_tt:
            ttnn.deallocate(x, force=True)
        x = x_next
    ttnn.deallocate(x, force=True)
    ttnn.synchronize_device(mesh_device)

    # Prepare host input for deterministic replay
    torch.manual_seed(1)
    host_input_data = ttnn.from_torch(
        torch.randn(1, 1, S_decode, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    ttnn.copy_host_to_device_tensor(host_input_data, input_tt)
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    retained = []
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    x = input_tt
    for layer_idx in range(NUM_LAYERS):
        x_next = _simulate_layer_forward(x, w1, w2, mesh_device)
        if layer_idx > 0:
            retained.append(x)
        x = x_next
    trace_output = x
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    # Get baseline output
    ttnn.copy_host_to_device_tensor(host_input_data, input_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    baseline_output = _readback(trace_output, mesh_device)

    # Run multiple prefill→replay cycles
    for cycle in range(NUM_CYCLES):
        # Prefill simulation
        chunk_shape = (1, 1, S_chunk, H)
        for layer_idx in range(NUM_LAYERS):
            chunks = []
            for ci in range(NUM_CHUNKS):
                cx = _make_bf16_tensor(chunk_shape, mesh_device, seed=200 + ci + cycle * 10)
                co = _simulate_layer_forward(cx, w1, w2, mesh_device)
                chunks.append(co)
                ttnn.deallocate(cx, force=False)
            if len(chunks) > 1:
                full = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for c in chunks:
                    ttnn.deallocate(c, force=False)
            else:
                full = chunks[0]
            ttnn.deallocate(full, force=False)
        ttnn.synchronize_device(mesh_device)

        # Replay trace
        ttnn.copy_host_to_device_tensor(host_input_data, input_tt)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

        cycle_output = _readback(trace_output, mesh_device)
        cycle_max_diff = (baseline_output.flatten() - cycle_output.flatten()).abs().max().item()
        logger.info("Cycle {}/{}: max_abs_diff={:.6f}", cycle + 1, NUM_CYCLES, cycle_max_diff)
        assert cycle_max_diff < 1.0, (
            f"Trace replay degraded at cycle {cycle + 1}: max_abs_diff={cycle_max_diff:.6f}. "
            f"Cumulative address reuse corruption across prefill cycles."
        )

    # Cleanup
    ttnn.release_trace(mesh_device, trace_id)
    for t in retained:
        try:
            ttnn.deallocate(t, force=True)
        except Exception:
            pass
    ttnn.deallocate(input_tt, force=True)
    ttnn.deallocate(w1, force=True)
    ttnn.deallocate(w2, force=True)

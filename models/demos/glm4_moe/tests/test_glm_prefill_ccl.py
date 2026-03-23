# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test that prefill all_reduce with device CCL (rs_ag) matches host fallback.

Validates that switching from host-side to device-side sync reduce_scatter + all_gather
for prefill all_reduce gives numerically equivalent results on BH Galaxy mesh (8,4).

Also validates that device CCL prefill does not corrupt subsequent trace capture/replay.

Run in Docker:
    docker compose ... run --rm vllm-tt bash -c \
      "cd /tt-metal && python -m pytest models/demos/glm4_moe/tests/test_glm_prefill_ccl.py -v"
"""

import os
import pytest
import torch
import ttnn
from loguru import logger

from models.demos.glm4_moe.tt.attention_tt import _simple_all_reduce
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_input_tensor(mesh_device, shape, seed=0):
    """Create a BF16 TILE_LAYOUT tensor replicated to all mesh devices."""
    torch.manual_seed(seed)
    torch_tensor = torch.randn(shape).bfloat16()
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return tt_tensor, torch_tensor


def _compute_golden_all_reduce(torch_tensor, mesh_shape, cluster_axis):
    """Compute expected all_reduce result for replicated input.

    When all devices in a reduction group hold the same tensor, the all_reduce
    result is group_size * tensor. This is because:
    - host impl: reads N devices, sums them, replicates result
    - rs_ag impl: reduce_scatter sums N partials, all_gather restores full dim
    Both produce N * original_tensor when inputs are identical.
    """
    group_size = mesh_shape[cluster_axis]
    return torch_tensor * group_size


# ---------------------------------------------------------------------------
# Test 1: rs_ag vs host correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["DP_axis0_8way", "TP_axis1_4way"])
@pytest.mark.parametrize("seq_len", [128, 512, 1024], ids=["S128", "S512", "S1024"])
def test_prefill_all_reduce_rs_ag_vs_host(mesh_device, cluster_axis, seq_len):
    """Verify rs_ag (device CCL) matches host fallback for prefill-shaped tensors.

    Creates two identical input tensors (since _simple_all_reduce deallocates input),
    runs one through impl="host" and the other through impl="rs_ag", and compares
    per-device outputs with PCC >= 0.999.

    Tests both:
    - cluster_axis=0 (DP reduce, 8-way across rows)
    - cluster_axis=1 (TP reduce, 4-way across columns)
    """
    shape = [1, 1, seq_len, 5120]
    mesh_shape = list(mesh_device.shape)
    num_devices = mesh_device.get_num_devices()
    group_size = mesh_shape[cluster_axis]

    logger.info(
        f"Testing rs_ag vs host: shape={shape}, axis={cluster_axis}, "
        f"mesh={mesh_shape}, group_size={group_size}"
    )

    # Create two identical input tensors (both impls deallocate their input)
    input_host, torch_input = _create_input_tensor(mesh_device, shape, seed=42)
    input_rsag, _ = _create_input_tensor(mesh_device, shape, seed=42)

    # Run host fallback (golden reference)
    result_host = _simple_all_reduce(
        input_host,
        mesh_device,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        impl="host",
    )

    # Run rs_ag (device-side sync CCL)
    result_rsag = _simple_all_reduce(
        input_rsag,
        mesh_device,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        impl="rs_ag",
    )

    ttnn.synchronize_device(mesh_device)

    # Extract per-device results
    host_devs = ttnn.get_device_tensors(result_host)
    rsag_devs = ttnn.get_device_tensors(result_rsag)

    host_torch = [ttnn.to_torch(t.cpu()) for t in host_devs]
    rsag_torch = [ttnn.to_torch(t.cpu()) for t in rsag_devs]

    # Also compute mathematical golden (group_size * input for replicated data)
    golden = _compute_golden_all_reduce(torch_input, mesh_shape, cluster_axis)

    # Verify rs_ag matches host on every device
    for i in range(num_devices):
        # Slice to logical shape (CCL may pad to tile boundaries)
        h = host_torch[i][..., :seq_len, :5120]
        r = rsag_torch[i][..., :seq_len, :5120]

        eq, pcc_str = comp_pcc(h, r, pcc=0.999)
        assert eq, (
            f"Device {i}: rs_ag != host. {pcc_str}. "
            f"axis={cluster_axis}, seq_len={seq_len}, mesh={mesh_shape}"
        )

    # Verify host result matches mathematical golden on device 0
    h0 = host_torch[0][..., :seq_len, :5120]
    eq_golden, golden_str = comp_pcc(golden, h0, pcc=0.998)
    assert eq_golden, (
        f"Host result != mathematical golden ({group_size}*input). {golden_str}. "
        f"axis={cluster_axis}, seq_len={seq_len}"
    )

    # Cleanup
    ttnn.deallocate(result_host, force=True)
    ttnn.deallocate(result_rsag, force=True)

    logger.info(f"PASS: rs_ag matches host on all {num_devices} devices")


# ---------------------------------------------------------------------------
# Test 2: rs_ag prefill does not corrupt subsequent trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1], ids=["TP_axis1"])
@pytest.mark.parametrize("seq_len", [128, 1024], ids=["S128", "S1024"])
def test_prefill_rs_ag_then_trace_replay(mesh_device, cluster_axis, seq_len):
    """Verify that running rs_ag prefill all_reduce does not leave device state
    that corrupts a subsequent trace capture + replay.

    This catches:
    - Leaked semaphore state from sync CCL ops
    - DRAM allocator corruption
    - Fabric state issues that cause trace replay to hang or produce wrong results

    Sequence:
    1. Run rs_ag all_reduce (simulating prefill)
    2. Sync device
    3. Capture a trace containing a simple matmul
    4. Execute the trace
    5. Verify matmul output is correct
    """
    num_devices = mesh_device.get_num_devices()
    prefill_shape = [1, 1, seq_len, 5120]

    # Step 1: Run rs_ag prefill all_reduce
    input_tensor, _ = _create_input_tensor(mesh_device, prefill_shape, seed=99)

    _ = _simple_all_reduce(
        input_tensor,
        mesh_device,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        impl="rs_ag",
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("Step 1: rs_ag prefill all_reduce completed")

    # Step 2: Set up trace tensors (decode-sized: [1,1,32,5120])
    decode_shape = [1, 1, 32, 5120]
    torch.manual_seed(123)
    t1_torch = torch.randn(decode_shape).bfloat16()
    t2_torch = torch.randn(decode_shape).bfloat16()

    t1 = ttnn.from_torch(
        t1_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    t2 = ttnn.from_torch(
        t2_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Compile the op first (required before trace capture)
    out = ttnn.add(t1, t2)
    ttnn.synchronize_device(mesh_device)
    logger.info("Step 2: Compile run completed")

    # Step 3: Capture trace
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = ttnn.add(t1, t2)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info("Step 3: Trace captured")

    # Step 4: Execute trace
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    logger.info("Step 4: Trace executed")

    # Step 5: Verify output
    expected = t1_torch + t2_torch
    actual_devs = ttnn.get_device_tensors(out)
    actual_torch = [ttnn.to_torch(t.cpu()) for t in actual_devs]

    for i in range(num_devices):
        eq, pcc_str = comp_pcc(expected, actual_torch[i], pcc=0.999)
        assert eq, (
            f"Device {i}: trace replay output corrupted after rs_ag prefill. {pcc_str}. "
            f"seq_len={seq_len}"
        )

    # Release trace
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"PASS: Trace replay correct on all {num_devices} devices after rs_ag prefill")


# ---------------------------------------------------------------------------
# Test 3: Multiple sequential rs_ag calls (simulating 92-layer prefill)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1], ids=["TP_axis1"])
def test_prefill_rs_ag_sequential_layers(mesh_device, cluster_axis):
    """Simulate multiple sequential rs_ag all_reduce calls (like 92 transformer layers).

    This catches:
    - Semaphore exhaustion after many sequential CCL ops
    - Memory leaks from internal semaphore allocation
    - DRAM fragmentation issues from repeated alloc/dealloc pattern

    Runs 10 sequential all_reduce calls (representative of multi-layer prefill)
    and verifies the last result is still correct.
    """
    shape = [1, 1, 128, 5120]
    mesh_shape = list(mesh_device.shape)
    num_devices = mesh_device.get_num_devices()
    group_size = mesh_shape[cluster_axis]
    num_layers = 10  # Representative subset of 92 layers

    logger.info(f"Testing {num_layers} sequential rs_ag calls, axis={cluster_axis}")

    for layer_idx in range(num_layers):
        input_tensor, torch_input = _create_input_tensor(
            mesh_device, shape, seed=layer_idx
        )
        result = _simple_all_reduce(
            input_tensor,
            mesh_device,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            impl="rs_ag",
        )
        # Only verify the last layer (avoid excessive D2H transfers)
        if layer_idx == num_layers - 1:
            ttnn.synchronize_device(mesh_device)
            golden = _compute_golden_all_reduce(torch_input, mesh_shape, cluster_axis)
            result_devs = ttnn.get_device_tensors(result)
            result_torch = [ttnn.to_torch(t.cpu()) for t in result_devs]

            for i in range(num_devices):
                r = result_torch[i][..., :128, :5120]
                eq, pcc_str = comp_pcc(golden, r, pcc=0.998)
                assert eq, (
                    f"Device {i}: layer {layer_idx} failed after {num_layers} "
                    f"sequential rs_ag calls. {pcc_str}"
                )
        else:
            ttnn.deallocate(result, force=True)

    logger.info(f"PASS: {num_layers} sequential rs_ag calls all correct")


# ---------------------------------------------------------------------------
# Test 4: rs_ag with EP_REDUCE_DEVICE pattern (both axes sequentially)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_prefill_rs_ag_both_axes_sequential(mesh_device):
    """Simulate the EP reduce pattern: TP reduce (axis=1) then DP reduce (axis=0).

    In the real model, decoder_layer_tt.py does:
    1. _simple_all_reduce(combined, cluster_axis=tp_axis)  # TP reduce
    2. _simple_all_reduce(result, cluster_axis=dp_axis)    # DP reduce

    This test verifies both axes work correctly in sequence.
    """
    shape = [1, 1, 128, 5120]
    mesh_shape = list(mesh_device.shape)
    num_devices = mesh_device.get_num_devices()
    tp_axis = 1  # BH Galaxy: TP on cols
    dp_axis = 0  # BH Galaxy: DP on rows

    logger.info(f"Testing sequential TP+DP rs_ag: mesh={mesh_shape}")

    # Step 1: TP reduce (axis=1, 4-way)
    input_tp, torch_input = _create_input_tensor(mesh_device, shape, seed=77)
    result_tp = _simple_all_reduce(
        input_tp,
        mesh_device,
        cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        impl="rs_ag",
    )
    ttnn.synchronize_device(mesh_device)

    # Step 2: DP reduce (axis=0, 8-way) on the TP-reduced result
    result_both = _simple_all_reduce(
        result_tp,
        mesh_device,
        cluster_axis=dp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        impl="rs_ag",
    )
    ttnn.synchronize_device(mesh_device)

    # Verify: result should be tp_group_size * dp_group_size * input = 4 * 8 * input = 32 * input
    expected_scale = mesh_shape[tp_axis] * mesh_shape[dp_axis]  # 4 * 8 = 32
    golden = torch_input * expected_scale

    result_devs = ttnn.get_device_tensors(result_both)
    result_torch = [ttnn.to_torch(t.cpu()) for t in result_devs]

    for i in range(num_devices):
        r = result_torch[i][..., :128, :5120]
        eq, pcc_str = comp_pcc(golden, r, pcc=0.997)
        assert eq, (
            f"Device {i}: sequential TP+DP reduce failed. {pcc_str}. "
            f"Expected scale={expected_scale}"
        )

    ttnn.deallocate(result_both, force=True)
    logger.info(f"PASS: Sequential TP+DP rs_ag correct (scale={expected_scale}x)")

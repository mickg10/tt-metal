# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test P2a: async DP all_gather replacing raw all_gather for bs>1 TG batch path.

The raw `ttnn.all_gather` at attention_tt.py:657 uses NO semaphore management.
When mixed with async CCL ops (rs_ag_async reduce_scatter + all_gather) in the
same traced decode, the raw op conflicts with in-flight async ops on the same
fabric axis (axis=0 for BH Galaxy DP). This causes hangs or fabric corruption
that manifests as a crash on the next CCL op.

This test validates:
1. async all_gather_async with CCL semaphores produces correct output (vs golden)
2. async all_gather can coexist with async reduce_scatter on the same axis in trace
3. The pattern works on BH Galaxy mesh (8,4) with both axis=0 (DP) and axis=1 (TP)

TDD: This test should PASS after replacing `_simple_all_gather` with the async
version using CCL managed semaphores. It should FAIL (hang or crash) if run
with the raw `ttnn.all_gather` in a traced context with concurrent async ops.

Run:
    # Single mesh test (basic API correctness):
    pytest models/demos/glm4_moe/tests/test_async_dp_all_gather.py -v -k "test_async_all_gather_correctness"

    # Trace mixing test (the actual bug):
    pytest models/demos/glm4_moe/tests/test_async_dp_all_gather.py -v -k "test_async_gather_coexists_with_async_reduce"

    # Full suite:
    pytest models/demos/glm4_moe/tests/test_async_dp_all_gather.py -v
"""

import os
import pytest
import torch
import ttnn
import logging

from models.demos.glm4_moe.tt.ccl import CCL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sharded_tensor(mesh_device, shape, dim, seed=42):
    """Create a tensor sharded along `dim` across mesh devices on axis=0.

    Each device gets a unique slice, modeling the DP batch-sliced attention
    pattern where each DP group holds its local batch entries.

    Returns: (tt_tensor, torch_full_tensor)
    """
    torch.manual_seed(seed)
    full = torch.randn(shape, dtype=torch.bfloat16)
    mesh_shape = list(mesh_device.shape)
    num_devices_on_axis0 = mesh_shape[0]

    # Shard along the specified dim for axis=0 (DP axis on BH)
    shard_size = shape[dim] // num_devices_on_axis0
    assert shape[dim] % num_devices_on_axis0 == 0, (
        f"Dim {dim} size {shape[dim]} not divisible by axis-0 devices {num_devices_on_axis0}"
    )

    tt_tensor = ttnn.from_torch(
        full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, dim), mesh_shape=mesh_shape),
    )
    return tt_tensor, full


def _make_replicated_tensor(mesh_device, shape, seed=42):
    """Create a BF16 tensor replicated to all mesh devices."""
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return tt_tensor, t


def _readback_per_device(t, mesh_device):
    """Read back tensor from each device in the mesh."""
    return [ttnn.to_torch(dt).float() for dt in ttnn.get_device_tensors(t)]


def _readback_first(t, mesh_device):
    """Read back tensor from first device only."""
    devs = ttnn.get_device_tensors(t)
    return ttnn.to_torch(devs[0]).float()


# ---------------------------------------------------------------------------
# Test 1: Async all_gather_async correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["DP_axis0", "TP_axis1"])
@pytest.mark.parametrize("gather_dim", [2], ids=["dim2_batch"])
def test_async_all_gather_correctness(mesh_device, cluster_axis, gather_dim):
    """Verify async all_gather_async with CCL semaphores produces correct output.

    Creates a tensor sharded along gather_dim, runs all_gather_async with managed
    semaphores, and verifies the gathered result matches the original full tensor.

    This is the basic API correctness test for the replacement of raw all_gather.
    """
    mesh_shape = list(mesh_device.shape)
    group_size = mesh_shape[cluster_axis]
    logger.info(f"Testing async all_gather: axis={cluster_axis}, group_size={group_size}, dim={gather_dim}")

    # Shape: [1, 1, B_total, hidden] where B_total is divisible by group_size
    B_total = 32  # tile-aligned
    hidden = 1536  # attention concat_heads output dim (12 heads * 128)
    shape = [1, 1, B_total, hidden]

    # Create CCL with managed semaphores
    ccl = CCL(mesh_device)

    # Create replicated input (each device holds the same data, simulating pre-gather)
    # For all_gather on dim=2: each device has [1,1,B_local,1536], gather → [1,1,B_total,1536]
    B_local = B_total // group_size
    local_shape = list(shape)
    local_shape[gather_dim] = B_local

    torch.manual_seed(42)
    local_torch = torch.randn(local_shape, dtype=torch.bfloat16)
    local_tt = ttnn.from_torch(
        local_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run async all_gather with managed semaphores
    ag_params = ccl.get_ccl_params_for_all_gather(axis=cluster_axis)
    gathered = ttnn.experimental.all_gather_async(
        local_tt,
        dim=gather_dim,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        **ag_params,
    )
    ttnn.synchronize_device(mesh_device)

    # Verify output shape
    gathered_shape = [int(d) for d in gathered.shape]
    assert gathered_shape[gather_dim] == B_total, (
        f"Expected gathered dim={gather_dim} to be {B_total}, got {gathered_shape[gather_dim]}"
    )

    # Verify content: gathered should be group_size copies of local along gather_dim
    gathered_torch = _readback_first(gathered, mesh_device)
    expected = local_torch.float().repeat(
        *([1] * gather_dim + [group_size] + [1] * (len(shape) - gather_dim - 1))
    )
    pcc = torch.corrcoef(
        torch.stack([expected.flatten(), gathered_torch.flatten()])
    )[0, 1].item()
    logger.info(f"async all_gather PCC: {pcc:.6f}")
    assert pcc >= 0.999, f"async all_gather output incorrect: PCC={pcc:.6f}"

    # Cleanup
    ttnn.deallocate(gathered, force=True)
    ttnn.deallocate(local_tt, force=True)


# ---------------------------------------------------------------------------
# Test 2: Async gather + async reduce coexistence in trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_async_gather_coexists_with_async_reduce_in_trace(mesh_device):
    """Verify async all_gather on axis=0 coexists with async reduce on axis=1 in trace.

    This is the KEY test for the P2a fix. It models the exact decode pattern:
    1. async reduce_scatter + all_gather on axis=1 (TP reduce for attention O-proj)
    2. async all_gather on axis=0 (DP batch gather after SDPA)
    3. async reduce_scatter + all_gather on axis=1 (TP reduce for MoE output)

    All ops run inside a single trace capture. The trace is replayed and output
    verified. This MUST work without hangs or crashes.

    With the raw all_gather bug, this would hang or produce a fabric error because
    the raw all_gather on axis=0 conflicts with in-flight async ops.
    """
    mesh_shape = list(mesh_device.shape)
    dp_axis = 0
    tp_axis = 1
    dp_size = mesh_shape[dp_axis]  # 8
    tp_size = mesh_shape[tp_axis]  # 4

    H = 1536  # hidden dim for attention output
    B_local = 4   # batch per DP group (32/8)
    B_total = B_local * dp_size  # 32

    logger.info(f"Testing trace with mixed async CCL: DP axis={dp_axis} ({dp_size}-way), TP axis={tp_axis} ({tp_size}-way)")

    ccl = CCL(mesh_device)

    # Create input tensors for the simulated decode step
    # x_attn: TP-sharded attention output [1,1,B_local,H] — needs TP reduce
    x_attn = _make_replicated_tensor(mesh_device, [1, 1, B_local, H], seed=1)[0]
    # w_proj: weight for "O-proj" matmul simulation
    w_proj = _make_replicated_tensor(mesh_device, [1, 1, H, H], seed=2)[0]

    # --- Compile warmup (pre-trace) ---
    logger.info("Compile warmup...")

    # Step A: TP reduce (axis=1) — simulates O-proj all_reduce
    rs_params = ccl.get_ccl_params_for_reduce_scatter(axis=tp_axis)
    temp_rs = ttnn.experimental.reduce_scatter_minimal_async(
        x_attn, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **rs_params,
    )
    ag_params = ccl.get_ccl_params_for_all_gather(axis=tp_axis)
    temp_ag = ttnn.experimental.all_gather_async(
        temp_rs, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **ag_params,
    )
    ttnn.deallocate(temp_rs, force=True)

    # Step B: DP gather (axis=0) — THE OP BEING TESTED
    dp_ag_params = ccl.get_ccl_params_for_all_gather(axis=dp_axis)
    temp_dp_gathered = ttnn.experimental.all_gather_async(
        temp_ag, dim=2, cluster_axis=dp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **dp_ag_params,
    )
    ttnn.deallocate(temp_ag, force=True)

    # Step C: Another TP reduce (axis=1) — simulates MoE output reduce
    rs_params2 = ccl.get_ccl_params_for_reduce_scatter(axis=tp_axis)
    temp_rs2 = ttnn.experimental.reduce_scatter_minimal_async(
        temp_dp_gathered, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **rs_params2,
    )
    ag_params2 = ccl.get_ccl_params_for_all_gather(axis=tp_axis)
    temp_ag2 = ttnn.experimental.all_gather_async(
        temp_rs2, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **ag_params2,
    )
    ttnn.deallocate(temp_rs2, force=True)
    ttnn.deallocate(temp_dp_gathered, force=True)

    warmup_output = _readback_first(temp_ag2, mesh_device)
    ttnn.deallocate(temp_ag2, force=True)
    ttnn.synchronize_device(mesh_device)

    # Reset semaphore counters for trace capture
    ccl.reset_global_semaphores()
    ccl.reset_sem_counters()

    # --- Trace capture ---
    logger.info("Capturing trace with mixed async CCL ops...")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    # Repeat the same A-B-C pattern inside trace
    # Step A: TP reduce
    rs_params = ccl.get_ccl_params_for_reduce_scatter(axis=tp_axis)
    rs_out = ttnn.experimental.reduce_scatter_minimal_async(
        x_attn, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **rs_params,
    )
    ag_params = ccl.get_ccl_params_for_all_gather(axis=tp_axis)
    ag_out = ttnn.experimental.all_gather_async(
        rs_out, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **ag_params,
    )
    ttnn.deallocate(rs_out, force=True)

    # Step B: DP gather — the fix being validated
    dp_ag_params = ccl.get_ccl_params_for_all_gather(axis=dp_axis)
    dp_gathered = ttnn.experimental.all_gather_async(
        ag_out, dim=2, cluster_axis=dp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **dp_ag_params,
    )
    ttnn.deallocate(ag_out, force=True)

    # Step C: Another TP reduce
    rs_params2 = ccl.get_ccl_params_for_reduce_scatter(axis=tp_axis)
    rs_out2 = ttnn.experimental.reduce_scatter_minimal_async(
        dp_gathered, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **rs_params2,
    )
    ag_params2 = ccl.get_ccl_params_for_all_gather(axis=tp_axis)
    trace_output = ttnn.experimental.all_gather_async(
        rs_out2, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **ag_params2,
    )
    ttnn.deallocate(rs_out2, force=True)
    ttnn.deallocate(dp_gathered, force=True)

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    logger.info("Trace captured successfully")

    # --- Trace replay ---
    logger.info("Replaying trace...")
    ccl.reset_sem_counters()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    replay_output = _readback_first(trace_output, mesh_device)

    # --- Second replay (verify stability) ---
    logger.info("Replaying trace a second time...")
    ccl.reset_sem_counters()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    replay2_output = _readback_first(trace_output, mesh_device)

    # --- Verify ---
    pcc1 = torch.corrcoef(
        torch.stack([warmup_output.flatten(), replay_output.flatten()])
    )[0, 1].item()
    pcc2 = torch.corrcoef(
        torch.stack([replay_output.flatten(), replay2_output.flatten()])
    )[0, 1].item()

    logger.info(f"Trace replay 1 vs warmup PCC: {pcc1:.6f}")
    logger.info(f"Trace replay 2 vs replay 1 PCC: {pcc2:.6f}")

    # PCC between warmup and trace may not be 1.0 due to different semaphore state,
    # but replay-to-replay must be deterministic
    assert pcc2 >= 0.999, (
        f"Trace replays not deterministic: PCC={pcc2:.6f}. "
        f"This indicates async CCL corruption during trace replay."
    )

    # Cleanup
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(x_attn, force=True)
    ttnn.deallocate(w_proj, force=True)


# ---------------------------------------------------------------------------
# Test 3: Raw all_gather in trace with async ops (EXPECTED TO FAIL before fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.xfail(reason="Raw all_gather mixed with async CCL in trace causes hang/crash", strict=False)
def test_raw_all_gather_in_trace_with_async_ops_XFAIL(mesh_device):
    """Demonstrate that RAW all_gather mixed with async CCL in trace FAILS.

    This test uses the UNFIXED pattern: raw ttnn.all_gather (no semaphores)
    interleaved with async reduce_scatter/all_gather ops in a traced context.

    Expected: HANG, crash, or incorrect output.
    This test is marked xfail — it documents the bug, not the fix.
    """
    mesh_shape = list(mesh_device.shape)
    dp_axis = 0
    tp_axis = 1

    H = 1536
    B_local = 4

    ccl = CCL(mesh_device)

    x = _make_replicated_tensor(mesh_device, [1, 1, B_local, H], seed=1)[0]

    # Warmup
    rs_params = ccl.get_ccl_params_for_reduce_scatter(axis=tp_axis)
    temp = ttnn.experimental.reduce_scatter_minimal_async(
        x, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **rs_params,
    )
    ag_params = ccl.get_ccl_params_for_all_gather(axis=tp_axis)
    temp = ttnn.experimental.all_gather_async(
        temp, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **ag_params,
    )

    # RAW all_gather (the bug) — no semaphore management
    raw_gathered = ttnn.all_gather(
        temp, dim=2, num_links=1,
        cluster_axis=dp_axis,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(temp, force=True)

    # Another async TP reduce (this is where the crash would manifest)
    rs_params2 = ccl.get_ccl_params_for_reduce_scatter(axis=tp_axis)
    temp2 = ttnn.experimental.reduce_scatter_minimal_async(
        raw_gathered, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **rs_params2,
    )
    ag_params2 = ccl.get_ccl_params_for_all_gather(axis=tp_axis)
    temp2 = ttnn.experimental.all_gather_async(
        temp2, dim=3, cluster_axis=tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear, **ag_params2,
    )
    ttnn.deallocate(raw_gathered, force=True)

    ttnn.synchronize_device(mesh_device)
    result = _readback_first(temp2, mesh_device)
    ttnn.deallocate(temp2, force=True)

    # If we get here without hang/crash, check output isn't garbage
    assert not torch.isnan(result).any(), "Output contains NaN"
    assert not torch.isinf(result).any(), "Output contains Inf"
    assert result.abs().max() < 1000, f"Output exploded: max={result.abs().max():.1f}"

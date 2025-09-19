# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# NOTE: To verify which mode (streaming vs non-streaming) is being used, look for log messages:
#   "SoftmaxBackward: Using NON-STREAMING kernel | Shape: ..."
#   "SoftmaxBackward: Using STREAMING kernel | Shape: ..."

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_dtype


def reference_softmax_backward_output(y: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    dot = (y * grad).sum(dim=axis, keepdim=True)
    return y * (grad - dot)


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    x = tensor1.to(torch.float32).reshape(-1)
    y = tensor2.to(torch.float32).reshape(-1)
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    vx = x - x_mean
    vy = y - y_mean
    num = torch.sum(vx * vy)
    den = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy)) + 1e-12
    return (num / den).item()


def assert_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor, pcc_threshold: float = 0.999) -> None:
    pcc_value = compute_pcc(tensor1, tensor2)
    assert pcc_value >= pcc_threshold, f"PCC {pcc_value:.6f} < threshold {pcc_threshold}"


def print_tolerance_metrics(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype_name: str = "", range: int = 0) -> None:
    """Calculate and print tolerance metrics between two tensors"""
    # Calculate actual differences
    abs_diff = torch.abs(tensor1 - tensor2)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()

    # Calculate relative difference
    rel_diff = abs_diff / (torch.abs(tensor2) + 1e-8)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Pearson correlation coefficient (PCC)
    pcc = compute_pcc(tensor1, tensor2)

    logger.info(f"\nTolerance metrics for {dtype_name} and range {range}:")
    logger.info(f"  Max absolute difference: {max_abs_diff:.6e}")
    logger.info(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    logger.info(f"  Max relative difference: {max_rel_diff:.6e}")
    logger.info(f"  Mean relative difference: {mean_rel_diff:.6e}")
    logger.info(f"  PCC: {pcc:.6f}")


BATCH_SIZE = 1
SEED = 77
PCC_THRESHOLD = 0.998
ABSOLUTE_TOLERANCE1 = 1e-2
ABSOLUTE_TOLERANCE4 = 4e-2
ABSOLUTE_TOLERANCE6 = 6e-3

# Relative tolerance is very sensitive to small values, because division by small value results in large relative difference.
# So we use absolute tolerance only.
RELATIVE_TOLERANCE = 0.0


@pytest.mark.parametrize(
    "input_shapes,atol,pcc_threshold",
    [
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 32]), ABSOLUTE_TOLERANCE1, PCC_THRESHOLD, id="1tile"),
        pytest.param(torch.Size([BATCH_SIZE, 3, 64, 64]), ABSOLUTE_TOLERANCE4, PCC_THRESHOLD, id="2tiles"),
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 9 * 32]), ABSOLUTE_TOLERANCE1, PCC_THRESHOLD, id="9tiles"),
        # Big tensor but small last row (4 tiles = 128 wide) - uses non-streaming mode
        pytest.param(
            torch.Size([BATCH_SIZE, 30, 6400, 5 * 32]), ABSOLUTE_TOLERANCE4, PCC_THRESHOLD, id="many_rows_5tiles"
        ),
        pytest.param(torch.Size([BATCH_SIZE, 2, 32, 63 * 32]), ABSOLUTE_TOLERANCE1, PCC_THRESHOLD, id="63tiles"),
        pytest.param(torch.Size([BATCH_SIZE, 3, 32, 64 * 32]), ABSOLUTE_TOLERANCE1, 0.98, id="64tiles"),
        pytest.param(torch.Size([BATCH_SIZE, 4, 32, 65 * 32]), ABSOLUTE_TOLERANCE1, 0.98, id="65tiles"),
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 127 * 32]), ABSOLUTE_TOLERANCE1, PCC_THRESHOLD, id="127tiles"),
        pytest.param(torch.Size([3, 1, 32, 128 * 32]), ABSOLUTE_TOLERANCE1, 0.94, id="128tiles"),
        pytest.param(torch.Size([BATCH_SIZE, 7, 128, 639 * 32]), ABSOLUTE_TOLERANCE1, 0.94, id="639tiles"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "range",
    [10],
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],  # only last dimension supported for now
)
def test_bw_softmax(input_shapes, atol, pcc_threshold, dtype, range, dim, device):
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -range, range, device, ttnn_dtype=dtype, seed=SEED)
    in_data, input_tensor = data_gen_with_range_dtype(
        input_shapes, -range, range, device, required_grad=True, ttnn_dtype=dtype, seed=SEED
    )

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    pt_softmax_tensor = torch.softmax(in_data, dim=dim, dtype=torch_dtype)
    tt_softmax_tensor = ttnn.from_torch(pt_softmax_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Test the fused kernel implementation
    tt_output_tensor_fused = ttnn.softmax_backward(tt_softmax_tensor, grad_tensor, dim=dim)
    pt_output_tensor_fused = ttnn.to_torch(tt_output_tensor_fused)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=dim)

    # Use torch.allclose with torch reference for bf16 and fp32 types
    if dtype in [ttnn.bfloat16, ttnn.float32]:
        try:
            assert torch.allclose(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                rtol=RELATIVE_TOLERANCE,
                atol=atol,
            )

            assert_pcc(pt_output_tensor_fused, pt_output_tensor_reference, pcc_threshold)
        except AssertionError:
            # Print detailed metrics on failure to help debug
            print_tolerance_metrics(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                dtype_name=f"dtype={dtype}",
                range=range,
            )
            raise


@pytest.mark.parametrize(
    "input_shapes,atol,pcc_threshold",
    [
        # Padded tensors with narrow rows - use non-streaming mode
        pytest.param(
            torch.Size([1, 1, 128, 14 * 32 + 2]), ABSOLUTE_TOLERANCE6, PCC_THRESHOLD, id="padded_14_first_face"
        ),
        pytest.param(
            torch.Size([2, 1, 64, 14 * 32 + 16]), ABSOLUTE_TOLERANCE6, PCC_THRESHOLD, id="padded_14_second_face1"
        ),
        pytest.param(
            torch.Size([2, 1, 64, 14 * 32 + 1]), ABSOLUTE_TOLERANCE6, PCC_THRESHOLD, id="padded_14_second_face2"
        ),
        # Padded tensors with wide rows - use streaming mode
        pytest.param(
            torch.Size([1, 5, 32, 300 * 32 + 3]), ABSOLUTE_TOLERANCE6, PCC_THRESHOLD, id="padded_300_first_face"
        ),
        pytest.param(
            torch.Size([7, 1, 64, 300 * 32 + 16]), ABSOLUTE_TOLERANCE6, PCC_THRESHOLD, id="padded_300_second_face1"
        ),
        pytest.param(
            torch.Size([3, 1, 32, 300 * 32 + 19]), ABSOLUTE_TOLERANCE4, PCC_THRESHOLD, id="padded_300_second_face2"
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "range",
    [10],
)
@pytest.mark.parametrize(
    "dim",
    [-1],  # test on last dimension
)
def test_bw_softmax_padded(input_shapes, atol, pcc_threshold, dtype, range, dim, device):
    """Test softmax backward with padded tensors (non-tile-aligned dimensions)"""

    torch.manual_seed(seed=SEED)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    grad_data = torch.rand(input_shapes, dtype=torch_dtype, requires_grad=False) * (range - (-range)) + (-range)
    in_data = torch.rand(input_shapes, dtype=torch_dtype, requires_grad=True) * (range - (-range)) + (-range)

    # Compute reference output on logical (unpadded) tensors
    pt_softmax_tensor = torch.softmax(in_data, dim=dim, dtype=torch_dtype)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=dim)

    # Use the pattern from test_fast_reduce_nc.py which works correctly
    # Create ttnn.Tensor directly, pad, then convert layout, then move to device
    tt_softmax_tensor = ttnn.Tensor(pt_softmax_tensor, dtype).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    tt_grad_tensor = ttnn.Tensor(grad_data, dtype).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    logger.debug(f"\nOriginal shape: {input_shapes}, Elements: {input_shapes.numel():,}")
    logger.debug(f"Padded shape: {tt_softmax_tensor.shape}")
    logger.debug(f"Tolerances: rtol={RELATIVE_TOLERANCE}, atol={atol}, pcc_threshold={pcc_threshold}")

    # Run softmax backward on padded tensors
    tt_output_tensor_fused = ttnn.softmax_backward(tt_softmax_tensor, tt_grad_tensor, dim=dim)

    # Convert back to torch (automatically unpads to logical shape)
    pt_output_tensor_fused = ttnn.to_torch(tt_output_tensor_fused)

    # Verify the output matches reference (only on logical/unpadded region)
    if dtype in [ttnn.bfloat16, ttnn.float32]:
        try:
            # The key test: output should match reference on the logical (unpadded) region
            # This verifies that padding did not corrupt the result
            assert (
                pt_output_tensor_fused.shape == pt_output_tensor_reference.shape
            ), f"Unpadded output shape mismatch: {pt_output_tensor_fused.shape} vs {pt_output_tensor_reference.shape}"

            assert torch.allclose(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                rtol=RELATIVE_TOLERANCE,
                atol=atol,
            ), f"Padded tensor output does not match reference! This means padding corrupted the reduction."
        except AssertionError:
            # Print detailed metrics on failure to help debug
            print_tolerance_metrics(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                dtype_name=f"dtype={dtype} (padded)",
                range=range,
            )
            raise


@pytest.mark.parametrize(
    "shape,expected_mode",
    [
        # Narrow rows (< 64 tiles) - use NON-STREAMING mode (tiles_per_block = width_tiles)
        ((1, 1, 32, 32), "NON-STREAMING"),  # 1 tile wide
        ((1, 1, 64, 64), "NON-STREAMING"),  # 2 tiles wide
        ((2, 2, 64, 128), "NON-STREAMING"),  # 4 tiles wide
        # Wide rows (>= 64 tiles) - use STREAMING mode (tiles_per_block = 1)
        ((4, 4, 128, 20480), "STREAMING"),  # 640 tiles wide
        ((8, 8, 64, 40960), "STREAMING"),  # 1280 tiles wide
    ],
)
def test_softmax_backward_kernel_selection(shape, expected_mode, device):
    """
    Test that verifies correct mode selection (streaming vs non-streaming) based on row width.

    Mode selection is based on whether the entire row fits in L1 cache (~1MB):
    - Rows < 64 tiles wide: NON-STREAMING mode (processes full row at once)
    - Rows >= 64 tiles wide: STREAMING mode (processes row in blocks of 1 tile)

    To see the mode selection logs, run with:
        export TT_METAL_LOGGER_LEVEL=2
        export TT_METAL_LOGGER_TYPES=Op

    Expected log output format:
        "SoftmaxBackward: Using NON-STREAMING kernel | Shape: 4x4 tiles (16 total) | Estimated L1: XX KB"
    or:
        "SoftmaxBackward: Using STREAMING kernel | Shape: 28x640 tiles (17920 total) | Estimated L1: XX KB"
    """
    logger.debug(f"\n{'='*40}")
    logger.debug(f"Testing mode selection for shape {shape}")
    logger.debug(f"Expected mode: {expected_mode}")
    logger.debug(f"{'='*40}")

    torch.manual_seed(SEED)

    # Create test tensors
    y = torch.softmax(torch.randn(shape, dtype=torch.bfloat16), dim=-1)
    grad = torch.randn(shape, dtype=torch.bfloat16)

    # Convert to ttnn
    tt_y = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_grad = ttnn.from_torch(grad, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Run operation - mode selection happens here
    # Check the logs to verify expected_mode was used
    logger.debug("⚠️  Check the logs above for mode selection message!")
    tt_output = ttnn.softmax_backward(tt_y, tt_grad, dim=-1)

    # Verify correctness
    pt_output = ttnn.to_torch(tt_output)
    pt_reference = reference_softmax_backward_output(y, grad, axis=-1)

    # Quick sanity check
    pcc = compute_pcc(pt_output, pt_reference)
    logger.debug(f"PCC: {pcc:.6f}")

    try:
        assert pcc >= PCC_THRESHOLD, f"Output doesn't match reference (PCC={pcc:.6f})"
    except AssertionError:
        # Print detailed metrics on failure to help debug
        print_tolerance_metrics(pt_output, pt_reference, dtype_name=f"shape={shape}", range=0)
        raise

    logger.debug(f"✅ Test passed for shape {shape} (expected {expected_mode} mode)")
    logger.debug(f"{'='*40}\n")

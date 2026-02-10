# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for HostInterface H2D/D2H socket loopback.

Sends 1024 bytes of data from host → device via H2D socket, the device kernel
copies it to the D2H socket, and the host reads it back. Verifies the data
matches end-to-end.

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io.py -v -s
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface


def test_host_io_loopback(device):
    """Test H2D → device loopback kernel → D2H with 1024-byte buffer, single iteration."""

    # Parameters
    fifo_size = 1024
    page_size = 1024
    data_size = 1024
    num_iterations = 1
    num_uint32 = data_size // 4  # 256 uint32 elements

    # Socket core: device (0,0), core (0,0)
    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    # Create H2D and D2H sockets on the same core
    logger.info(f"Creating H2D + D2H sockets on core {core_coord}, fifo_size={fifo_size}")
    h2d_socket = ttnn.H2DSocket(device, socket_core, ttnn.BufferType.L1, fifo_size, ttnn.H2DMode.HOST_PUSH)
    d2h_socket = ttnn.D2HSocket(device, socket_core, fifo_size)

    # Create input host tensor with known data pattern: 0, 1, 2, ..., 255
    torch_input = torch.arange(num_uint32, dtype=torch.int32).reshape(1, num_uint32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Create output host tensor (pre-allocated, zeroed)
    torch_output = torch.zeros(1, num_uint32, dtype=torch.int32)
    output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Launch loopback kernel via HostInterface op (non-blocking enqueue)
    logger.info("Launching loopback kernel on device...")
    host_io = HostInterface()
    host_io.op(
        h2d_socket,
        d2h_socket,
        page_size,
        io_tensors,
        num_iterations=num_iterations,
        data_size=data_size,
        pull_from_host=False,
    )

    # Write input tensor to device via H2D socket
    logger.info(f"Writing {data_size} bytes to H2D socket...")
    h2d_socket.write_tensor(input_tensor)

    # Read output tensor from device via D2H socket
    logger.info(f"Reading {data_size} bytes from D2H socket...")
    d2h_socket.read_tensor(output_tensor)

    # Wait for all socket transfers to complete
    h2d_socket.barrier()
    d2h_socket.barrier()

    # Compare input and output
    result_torch = ttnn.to_torch(output_tensor)
    logger.info(f"Input  (first 8): {torch_input[0, :8].tolist()}")
    logger.info(f"Output (first 8): {result_torch[0, :8].tolist()}")

    match = torch.equal(torch_input, result_torch)
    logger.info(f"Data match: {match}")
    assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"
    logger.info("Host I/O loopback test passed!")

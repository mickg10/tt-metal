# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for HostInterface H2D/D2H socket loopback.

"""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface


@pytest.mark.parametrize(
    "tensor_size_bytes, fifo_size, num_iterations",
    [
        (32, 512, 256),
        (64, 1024, 256),
        (64, 256, 256),
        (1024, 2048, 32),
    ],
)
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
        ttnn.H2DMode.DEVICE_PULL,
    ],
)
def test_host_io_loopback(mesh_device, tensor_size_bytes, fifo_size, num_iterations, h2d_mode):
    fifo_size = 1024
    tensor_size_bytes = 64
    tensor_size_datums = tensor_size_bytes // 4
    num_iterations = 128

    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    logger.info("Creating and Running Host Interface")
    h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size, h2d_mode)
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size)
    host_io = HostInterface(h2d_socket, d2h_socket, tensor_size_bytes)
    host_io.run()

    logger.info("Transferring Data Over H <-> D Interface for {num_iterations} iterations")

    for i in range(num_iterations):
        torch_input = torch.arange(i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.int32).reshape(
            1, tensor_size_datums
        )
        input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.int32)
        output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        h2d_socket.write_tensor(input_tensor)
        d2h_socket.read_tensor(output_tensor)

        result_torch = ttnn.to_torch(output_tensor)
        match = torch.equal(torch_input, result_torch)
        assert match, f"H2D → D2H loopback data mismatch!\nExpected: {torch_input}\nGot: {result_torch}"

    h2d_socket.barrier()
    d2h_socket.barrier()
    host_io.terminate()

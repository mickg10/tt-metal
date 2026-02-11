# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


class HostInterface:
    def __init__(self, h2d_socket, d2h_socket, page_size):
        self.h2d_socket = h2d_socket
        self.d2h_socket = d2h_socket
        self.page_size = page_size
        self.h2d_socket.set_page_size(self.page_size)
        self.d2h_socket.set_page_size(self.page_size)

        if len(h2d_socket.get_active_cores()) != 1 or len(d2h_socket.get_active_cores()) != 1:
            raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
        if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")
        if h2d_socket.get_mesh_device() != d2h_socket.get_mesh_device():
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same mesh device.")

        self.mesh_core_coord = self.h2d_socket.get_active_cores()[0]
        self.termination_semaphore = ttnn.create_global_semaphore(
            h2d_socket.get_mesh_device(),
            ttnn.CoreRangeSet([ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]),
            0,
            ttnn.BufferType.L1,
        )

    def run(self):
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.h2d_socket.get_mesh_device()
        )

        socket_kernel_ct_args = [
            self.h2d_socket.get_config_buffer_address(),
            self.d2h_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.page_size,
            self.h2d_socket.get_h2d_mode() == ttnn.H2DMode.DEVICE_PULL,
        ]
        kernel = ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
            ),
            compile_time_args=socket_kernel_ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[],
        )
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        mesh_program_descriptor[
            ttnn.MeshCoordinateRange(self.mesh_core_coord.device_coord, self.mesh_core_coord.device_coord)
        ] = program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]
        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def terminate(self):
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)

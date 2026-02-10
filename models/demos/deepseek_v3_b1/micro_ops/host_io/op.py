# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import ttnn


class HostInterface:
    @staticmethod
    def golden():
        # Unimplemented for now, since this op simply manages H2D and D2H communication
        pass

    def op(
        self,
        h2d_socket,
        d2h_socket,
        page_size,
        termination_tensor,
    ):
        h2d_socket.set_page_size(page_size)
        d2h_socket.set_page_size(page_size)

        if len(h2d_socket.get_active_cores()) != 1 or len(d2h_socket.get_active_cores()) != 1:
            raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
        if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")
        if h2d_socket.get_mesh_device() != d2h_socket.get_mesh_device():
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same mesh device.")

        mesh_core_coord = h2d_socket.get_active_cores()[0]

        socket_kernel_ct_args = [
            h2d_socket.get_config_buffer_address(),
            d2h_socket.get_config_buffer_address(),
            page_size,
            data_size,
            num_iterations,
            int(pull_from_host),
        ]

        kernel = ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(mesh_core_coord.core_coord),
            compile_time_args=socket_kernel_ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[],
        )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        mesh_program_descriptor[ttnn.MeshCoordinateRange(mesh_core_coord.device_coord)] = program

        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

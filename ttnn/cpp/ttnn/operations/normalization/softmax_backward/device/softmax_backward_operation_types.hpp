// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/kernel_types.hpp>

#include <cstdint>

namespace ttnn::prim {

struct SoftmaxBackwardParams {
    const uint32_t dim;
};

struct SoftmaxBackwardInputs {
    ttnn::Tensor softmax_output;
    ttnn::Tensor upstream_grad;
};

// Shared variables used by both non-streaming and streaming factories
// Only stores kernel handles needed for override_runtime_arguments
struct shared_variables_t {
    tt::tt_metal::KernelHandle unary_reader_kernel_id;
    tt::tt_metal::KernelHandle unary_writer_kernel_id;
};

}  // namespace ttnn::prim

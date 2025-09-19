// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_backward_operation_types.hpp"

#include <tt-metalium/kernel_types.hpp>
#include <ttnn/device_operation.hpp>

namespace ttnn::prim {

struct SoftmaxBackwardFactory {
    using shared_variables_t = shared_variables_t;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SoftmaxBackwardParams& operation_attributes,
        const SoftmaxBackwardInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SoftmaxBackwardParams& operation_attributes,
        const SoftmaxBackwardInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

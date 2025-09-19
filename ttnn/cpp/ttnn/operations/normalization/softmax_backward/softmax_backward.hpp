// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {
/**
 * @brief Executes the backpropagation on softmax operation on a tensor along a specified dimension.
 *
 * Computes softmax_backward(y, grad, dim) = y * (grad - (y * grad).sum(dim, keepdim=True)) along the specified
 * dimension. The operation creates a new output tensor.
 */
ttnn::Tensor softmax_backward(const ttnn::Tensor& softmax_output_tensor, const ttnn::Tensor& grad_tensor, int32_t dim);
}  // namespace ttnn::operations::normalization

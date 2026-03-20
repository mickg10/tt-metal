// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that paged_update_cache actually modifies the cache on Blackhole Galaxy mesh.
// Bug: paged_update_cache is a silent no-op on BH Galaxy mesh (32 devices).
// The untilize->modify->retilize pipeline runs but the DRAM write-back is silently dropped.
// Single device works. Multi-device mesh does NOT write.
// Related issues: #14594 (BH LLK untilize), #27193 (BH SDPA decode skip)

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>
#include <set>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache.hpp"
#include "ttnn_test_fixtures.hpp"

using namespace tt::tt_metal;

namespace ttnn::test {

// Helper: create a HEIGHT_SHARDED input tensor for paged_update_cache.
//
// paged_update_cache requires the input tensor to be HEIGHT_SHARDED on L1.
// The shape is [1, batch, num_heads, head_dim] with heads padded to tile height (32).
//
// This follows the exact pattern from the upstream nightly Python tests
// (test_paged_update_cache.py:run_test_update_cache_decode):
//   1. Create host data with shape [1, batch, padded_heads, head_dim]
//   2. Create tensor from vector with TILE layout
//   3. Move to device with HEIGHT_SHARDED L1 memory config
static ttnn::Tensor make_sharded_input(
    distributed::MeshDevice& device,
    uint32_t batch,
    uint32_t num_heads,
    uint32_t head_dim,
    float fill_value) {
    const uint32_t padded_heads = ((num_heads + 31) / 32) * 32;  // pad to tile height

    // Logical shape for the operation: [1, batch, num_heads, head_dim]
    // Physical (padded) shape for tiling: [1, batch, padded_heads, head_dim]
    const uint32_t vol = 1 * batch * padded_heads * head_dim;

    // Fill with bfloat16 values: actual heads get fill_value, padding gets 0
    std::vector<bfloat16> host_data(vol, bfloat16(0.0f));
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t h = 0; h < num_heads; h++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                uint32_t idx = b * padded_heads * head_dim + h * head_dim + d;
                host_data[idx] = bfloat16(fill_value);
            }
        }
    }

    // Create host tensor with padded shape, TILE layout
    ttnn::Shape padded_shape({1, batch, padded_heads, head_dim});
    TensorSpec host_spec(
        padded_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    auto host_tensor = Tensor::from_vector(host_data, host_spec);

    // Reshape to logical shape (keeps the padded backing but changes logical dimensions)
    ttnn::Shape logical_shape({1, batch, num_heads, head_dim});
    host_tensor = host_tensor.reshape(logical_shape);

    // Compute shard dimensions for HEIGHT_SHARDED
    // shard_height = total_physical_height / num_cores, shard_width = padded_head_dim
    // total physical volume / last_dim / num_cores = shard_height
    uint32_t num_cores = batch;
    uint32_t physical_last_dim = host_tensor.padded_shape()[-1];
    uint32_t shard_height = host_tensor.physical_volume() / physical_last_dim / num_cores;
    uint32_t shard_width = physical_last_dim;

    CoreCoord grid_size = device.compute_with_storage_grid_size();
    CoreRangeSet shard_grid = num_cores_to_corerangeset(num_cores, grid_size, /*row_wise=*/true);

    ShardSpec shard_spec(shard_grid, {shard_height, shard_width}, ShardOrientation::ROW_MAJOR);
    MemoryConfig mem_cfg(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);

    return ttnn::to_device(host_tensor, &device, mem_cfg);
}

// Helper: create an INT32 tensor on device from a vector of int32_t values.
// Used for page_table and update_idxs tensors.
static ttnn::Tensor make_int32_tensor_on_device(
    distributed::MeshDevice& device,
    const std::vector<int32_t>& data,
    const ttnn::Shape& shape) {
    TensorSpec spec(
        shape,
        TensorLayout(DataType::INT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    auto host_tensor = Tensor::from_vector(data, spec);
    return ttnn::to_device(host_tensor, &device, std::nullopt);
}

class PagedUpdateCacheBHTest : public TTNNFixtureWithDevice {};

// Test: write a known value via paged_update_cache (non-paged path) and verify the cache changes.
// Uses the non-paged code path (no page_table): cache_shape = [batch, num_heads, max_seq_len, head_dim]
// with update_idxs_tensor for position indices.
TEST_F(PagedUpdateCacheBHTest, SinglePositionUpdate) {
    auto& device = *device_;

    const uint32_t num_heads = 1;      // Keep small; must pad to 32 for tile
    const uint32_t head_dim = 128;
    const uint32_t max_seq_len = 2048;
    const uint32_t batch = 1;          // Single user for simplicity
    const uint32_t update_pos = 5;     // Position to write within the sequence
    const float known_value = 42.0f;   // Value to write (easy to verify)

    // 1. Create cache tensor — all zeros, shape [batch, num_heads, max_seq_len, head_dim]
    // Non-paged: dim0 = batch (matches input dim1)
    auto cache_shape = ttnn::Shape({batch, num_heads, max_seq_len, head_dim});
    auto cache = ttnn::zeros(cache_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    // 2. Create HEIGHT_SHARDED input tensor with known_value
    auto input = make_sharded_input(device, batch, num_heads, head_dim, known_value);

    // 3. Create update_idxs tensor — [batch] INT32 values, DRAM INTERLEAVED
    std::vector<int32_t> update_idxs_data = {static_cast<int32_t>(update_pos)};
    auto update_idxs = make_int32_tensor_on_device(device, update_idxs_data, ttnn::Shape({batch}));

    // 4. Read cache BEFORE update — should be zeros
    {
        auto cache_host = ttnn::from_device(cache);
        auto cache_vec = cache_host.to_vector<bfloat16>();
        // Index into [batch=0, head=0, pos=update_pos, dim=0]
        uint32_t before_idx = 0 * num_heads * max_seq_len * head_dim +
                              0 * max_seq_len * head_dim +
                              update_pos * head_dim +
                              0;
        float before_val = static_cast<float>(cache_vec[before_idx]);
        EXPECT_NEAR(before_val, 0.0f, 0.01f) << "Cache should be zeros before update";
    }

    // 5. Run paged_update_cache (non-paged path: no page_table)
    ttnn::experimental::paged_update_cache(
        cache,
        input,
        std::vector<uint32_t>{},  // update_idxs vector (empty — using tensor)
        update_idxs,      // update_idxs_tensor
        std::nullopt,     // share_cache
        std::nullopt,     // page_table (nullopt = non-paged)
        uint32_t(0),      // batch_offset
        std::nullopt,     // compute_kernel_config
        std::nullopt      // mesh_coords
    );

    // 6. Read cache AFTER update
    auto cache_after_host = ttnn::from_device(cache);
    auto cache_after_vec = cache_after_host.to_vector<bfloat16>();

    uint32_t after_idx = 0 * num_heads * max_seq_len * head_dim +
                         0 * max_seq_len * head_dim +
                         update_pos * head_dim +
                         0;
    float after_val = static_cast<float>(cache_after_vec[after_idx]);

    // 7. Verify: cache at update_pos should now have known_value, not zeros
    std::cout << "Cache BEFORE at pos=" << update_pos << ": 0.0" << std::endl;
    std::cout << "Cache AFTER  at pos=" << update_pos << ": " << after_val << std::endl;
    std::cout << "Expected: " << known_value << std::endl;

    // The bug: on BH Galaxy mesh, after_val == 0 (no-op)
    // Expected: after_val == known_value (42.0)
    EXPECT_NE(after_val, 0.0f) << "paged_update_cache had NO EFFECT — cache unchanged (BH bug!)";
    EXPECT_NEAR(after_val, known_value, 1.0f) << "paged_update_cache wrote wrong value";

    // Also verify a position that was NOT updated is still zero
    uint32_t untouched_idx = 0 * num_heads * max_seq_len * head_dim +
                             0 * max_seq_len * head_dim +
                             (update_pos + 1) * head_dim +
                             0;
    float untouched_val = static_cast<float>(cache_after_vec[untouched_idx]);
    EXPECT_NEAR(untouched_val, 0.0f, 0.01f) << "Untouched cache position should still be zero";
}

// Test: paged path with page_table — write to a specific block via indirection.
TEST_F(PagedUpdateCacheBHTest, PagedSinglePositionUpdate) {
    auto& device = *device_;

    const uint32_t num_heads = 1;
    const uint32_t head_dim = 128;
    const uint32_t block_size = 64;
    const uint32_t num_blocks = 4;
    const uint32_t batch = 1;
    const uint32_t update_pos = 5;     // Global position — maps to block 0, offset 5
    const float known_value = 42.0f;

    // 1. Create cache tensor — paged: [num_blocks, num_heads, block_size, head_dim]
    auto cache_shape = ttnn::Shape({num_blocks, num_heads, block_size, head_dim});
    auto cache = ttnn::zeros(cache_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    // 2. Create HEIGHT_SHARDED input tensor
    auto input = make_sharded_input(device, batch, num_heads, head_dim, known_value);

    // 3. Create page_table — identity mapping: [batch, max_num_blocks_per_seq]
    // page_table shape: [batch, num_blocks] where page_table[b][i] = physical block for virtual block i
    // With identity mapping, virtual block i maps to physical block i.
    const uint32_t max_num_blocks_per_seq = num_blocks;
    std::vector<int32_t> page_table_data(batch * max_num_blocks_per_seq);
    for (uint32_t i = 0; i < max_num_blocks_per_seq; i++) {
        page_table_data[i] = static_cast<int32_t>(i);
    }
    auto page_table = make_int32_tensor_on_device(
        device, page_table_data, ttnn::Shape({batch, max_num_blocks_per_seq}));

    // 4. Create update_idxs tensor — global position [batch]
    std::vector<int32_t> update_idxs_data = {static_cast<int32_t>(update_pos)};
    auto update_idxs = make_int32_tensor_on_device(device, update_idxs_data, ttnn::Shape({batch}));

    // 5. Run paged_update_cache (paged path: with page_table)
    ttnn::experimental::paged_update_cache(
        cache,
        input,
        std::vector<uint32_t>{},  // update_idxs vector (empty — using tensor)
        update_idxs,      // update_idxs_tensor
        std::nullopt,     // share_cache
        page_table,       // page_table
        uint32_t(0),      // batch_offset
        std::nullopt,     // compute_kernel_config
        std::nullopt      // mesh_coords
    );

    // 6. Read cache back and verify
    auto cache_host = ttnn::from_device(cache);
    auto cache_vec = cache_host.to_vector<bfloat16>();

    // update_pos=5 is in block 0 (5 / 64 = 0), offset 5 (5 % 64 = 5)
    // With identity page_table, physical block 0 = virtual block 0
    // Index: block * (num_heads * block_size * head_dim) + head * (block_size * head_dim) + offset * head_dim + d
    uint32_t block_idx = update_pos / block_size;
    uint32_t offset_in_block = update_pos % block_size;
    uint32_t target_idx = block_idx * num_heads * block_size * head_dim +
                          0 * block_size * head_dim +
                          offset_in_block * head_dim +
                          0;
    float result = static_cast<float>(cache_vec[target_idx]);

    std::cout << "Paged: pos=" << update_pos
              << " block=" << block_idx
              << " offset=" << offset_in_block
              << " expected=" << known_value
              << " got=" << result << std::endl;

    EXPECT_NE(result, 0.0f) << "paged_update_cache had NO EFFECT at pos=" << update_pos;
    EXPECT_NEAR(result, known_value, 1.0f) << "paged_update_cache wrote wrong value";

    // Verify an untouched position is still zero (e.g., block 1, offset 0)
    uint32_t untouched = 1 * num_heads * block_size * head_dim +
                         0 * block_size * head_dim +
                         0 * head_dim +
                         0;
    float untouched_val = static_cast<float>(cache_vec[untouched]);
    EXPECT_NEAR(untouched_val, 0.0f, 0.01f) << "Untouched block should still be zero";
}

// Test: non-paged path, multiple positions including tile boundaries.
TEST_F(PagedUpdateCacheBHTest, TileBoundaryPositions) {
    auto& device = *device_;

    const uint32_t num_heads = 1;
    const uint32_t head_dim = 128;
    const uint32_t max_seq_len = 2048;
    const uint32_t batch = 1;

    // Positions to test — include tile boundaries (multiples of 32)
    std::vector<uint32_t> test_positions = {0, 15, 31, 32, 47, 63};

    for (uint32_t pos : test_positions) {
        float known_val = static_cast<float>(pos + 1);  // unique per position

        // Fresh cache (zeros)
        auto cache_shape = ttnn::Shape({batch, num_heads, max_seq_len, head_dim});
        auto cache = ttnn::zeros(cache_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

        // Input with known value
        auto input = make_sharded_input(device, batch, num_heads, head_dim, known_val);

        // Update indices
        std::vector<int32_t> idx_data = {static_cast<int32_t>(pos)};
        auto idx = make_int32_tensor_on_device(device, idx_data, ttnn::Shape({batch}));

        // Update (non-paged path)
        ttnn::experimental::paged_update_cache(
            cache, input,
            std::vector<uint32_t>{},  // update_idxs vector (empty)
            idx,                       // update_idxs_tensor
            std::nullopt,              // share_cache
            std::nullopt,              // page_table (non-paged)
            uint32_t(0),               // batch_offset
            std::nullopt,              // compute_kernel_config
            std::nullopt               // mesh_coords
        );

        // Read back and verify
        auto cache_host = ttnn::from_device(cache);
        auto cache_vec = cache_host.to_vector<bfloat16>();

        uint32_t target_idx = 0 * num_heads * max_seq_len * head_dim +
                              0 * max_seq_len * head_dim +
                              pos * head_dim +
                              0;
        float result = static_cast<float>(cache_vec[target_idx]);

        std::cout << "pos=" << pos << ": expected=" << known_val << " got=" << result << std::endl;
        EXPECT_NE(result, 0.0f) << "paged_update_cache had NO EFFECT at pos=" << pos;
        EXPECT_NEAR(result, known_val, 1.0f) << "paged_update_cache wrote wrong value at pos=" << pos;
    }
}

}  // namespace ttnn::test

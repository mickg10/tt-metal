// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel_defs.h"
#include "tt-metalium/constants.hpp"
#include "api/debug/dprint_pages.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t first_physical_x = get_named_compile_time_arg_val("first_physical_x");
    constexpr uint32_t first_physical_y = get_named_compile_time_arg_val("first_physical_y");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto partial_semaphore = get_arg_val<uint32_t>(argidx++);
    const auto is_send_core = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto raw_scores_semaphore = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5 = tt::CBIndex::c_7;
    constexpr auto cb_w2c_in6 = tt::CBIndex::c_8;
    constexpr auto cb_w2c_in7 = tt::CBIndex::c_9;

    // Aliases
    constexpr auto cb_w2c_in8 = tt::CBIndex::c_6;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w);
    constexpr uint32_t out_tile_size = get_tile_size(cb_s2c_out);

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = is_send_core ? 2 * 72 : 2 * 76;
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(partial_semaphore);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    const uint64_t partial_semaphore_noc_addr1 =
        get_noc_addr(neighbor1_physical_x, neighbor1_physical_y, semaphore_addr);

    const uint64_t partial_semaphore_noc_addr2 =
        get_noc_addr(neighbor2_physical_x, neighbor2_physical_y, semaphore_addr);

    const uint32_t local_src_addr = get_write_ptr(cb_s2c_out);
    const uint32_t local_dst_addr = get_write_ptr(cb_w2c_in2);
    const uint64_t neighbor_dst_addr1 = get_noc_addr(neighbor1_physical_x, neighbor1_physical_y, local_dst_addr);
    const uint64_t neighbor_dst_addr2 = get_noc_addr(neighbor2_physical_x, neighbor2_physical_y, local_dst_addr);

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    constexpr uint32_t COLLECTOR_CORE_ID = 7;

    const uint32_t local_collector_addr = 1024 + get_write_ptr(cb_w2c_in4);
    const uint64_t collector_dst_base_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, local_collector_addr);
    const uint32_t collector_offset = core_id * out_tile_size;

    constexpr uint32_t group_score_size = tt::constants::FACE_WIDTH * sizeof(uint16_t);

    //-------------------------------------------------------------------------
    // Group scores (7 cores -> 1 collector core)
    //-------------------------------------------------------------------------

    // Group scores: data exists in 1st and 4th row of the first face of the bfloat16 tile
    constexpr uint32_t group_score_offset1 = 0;
    constexpr uint32_t group_score_offset2 = 4 * group_score_size;
    const uint32_t local_group_score_base_addr = get_write_ptr(cb_c2w_rdy);
    const uint32_t local_group_score_src_addr1 = local_group_score_base_addr + group_score_offset1;
    const uint32_t local_group_score_src_addr2 = local_group_score_base_addr + group_score_offset2;

    const uint32_t local_group_scores_dst_addr1 = local_collector_addr + core_id * group_score_size;
    const uint32_t local_group_scores_dst_addr2 = local_group_scores_dst_addr1 + 8 * group_score_size;

    const uint64_t collector_semaphore_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, semaphore_addr);

    //-------------------------------------------------------------------------
    // Group masks (1 collector core -> 7 cores)
    //-------------------------------------------------------------------------
    const uint32_t local_group_masks_addr = get_write_ptr(cb_w2c_in5);
    const uint64_t group_masks_noc_addr = get_noc_multicast_addr(
        first_physical_x, first_physical_y, collector_physical_x, collector_physical_y, local_group_masks_addr);
    const uint64_t group_semaphore_noc_addr = get_noc_multicast_addr(
        first_physical_x, first_physical_y, collector_physical_x, collector_physical_y, semaphore_addr);

    //-------------------------------------------------------------------------
    // Top8 partials (7 cores -> 1 collector core)
    //-------------------------------------------------------------------------
    constexpr uint32_t partials_size = 2 * tt::constants::FACE_HW * sizeof(uint16_t);  // values and indices

    // Top8 partials: data exists in first 16 rows
    const uint32_t local_top8_partials_src_addr = get_write_ptr(cb_w2c_in8);

    const uint32_t local_top8_dst_base_addr = get_write_ptr(cb_w2c_in6);
    const uint32_t local_top8_dst_addr = local_top8_dst_base_addr + core_id * partials_size;

    //-------------------------------------------------------------------------
    // Raw scores (7 cores -> 1 collector core)
    //-------------------------------------------------------------------------
    constexpr uint32_t raw_scores_size = tt::constants::TILE_HW * sizeof(uint16_t);

    uint32_t raw_scores_semaphore_addr = get_semaphore(raw_scores_semaphore);
    volatile tt_l1_ptr uint32_t* my_raw_scores_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_scores_semaphore_addr);
    *my_raw_scores_semaphore_ptr = 0;

    // Top8 partials: data exists in first 16 rows
    const uint32_t local_raw_scores_src_addr = get_write_ptr(cb_w2c_in3);

    const uint32_t local_raw_scores_dst_base_addr = get_write_ptr(cb_w2c_in7);
    const uint32_t local_raw_scores_dst_addr = local_raw_scores_dst_base_addr + core_id * raw_scores_size;

    const uint64_t raw_scores_semaphore_noc_addr =
        get_noc_addr(collector_physical_x, collector_physical_y, raw_scores_semaphore_addr);
    const uint64_t raw_scores_noc_addr = get_noc_multicast_addr(
        first_physical_x, first_physical_y, collector_physical_x, collector_physical_y, local_raw_scores_dst_base_addr);

    //-------------------------------------------------------------------------
    *my_semaphore_ptr = 0;

    if (is_send_core) {
        // Since neighbor2 is farther, we send it first.
        // Set state for the writes
        noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_dst_addr2, out_tile_size, /*noc=*/1, vchannel);

        // Wait for the data1 to be ready
        cb_wait_front(cb_c2w_rdy, 1);

        // Send the data to the neighbor1
        noc_async_write_one_packet_with_state</*posted=*/true>(local_src_addr, neighbor_dst_addr2);

        // Signal neighbor1 that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(partial_semaphore_noc_addr2, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        cb_pop_front(cb_c2w_rdy, 1);

        noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_dst_addr1, out_tile_size, /*noc=*/1, vchannel);

        // Wait for the data2 to be ready
        cb_wait_front(cb_c2w_rdy, 1);

        // Send the data to the neighbor2
        noc_async_write_one_packet_with_state</*posted=*/true>(local_src_addr, neighbor_dst_addr1);

        // Signal neighbor2 that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(partial_semaphore_noc_addr1, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        cb_pop_front(cb_c2w_rdy, 1);

        return;
    }

    // -------------------------------------------------------------------------
    // Rest of the 8 cores do more
    // -------------------------------------------------------------------------
    cb_reserve_back(cb_w2c_in2, 1);

    // Wait for the data from sender cores to be ready
    noc_semaphore_wait_min(my_semaphore_ptr, 1);
    *my_semaphore_ptr = 0;
    cb_push_back(cb_w2c_in2, 1);

    // Wait for group scores to be ready from compute
    cb_wait_front(cb_c2w_rdy, 1);

    //-------------------------------------------------------------------------
    // Cores sending data to the collector core
    //-------------------------------------------------------------------------
    if (core_id != COLLECTOR_CORE_ID) {
        noc_async_write_one_packet_set_state</*posted=*/true>(
            collector_dst_base_addr, group_score_size, /*noc=*/1, vchannel);

        // Send them over to the collector core
        noc_async_write_one_packet_with_state</*posted=*/true>(
            local_group_score_src_addr1, local_group_scores_dst_addr1);
        noc_async_write_one_packet_with_state</*posted=*/true>(
            local_group_score_src_addr2, local_group_scores_dst_addr2);

        // Signal the collector core that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        // We are done with the group scores
        cb_pop_front(cb_c2w_rdy, 1);

        cb_reserve_back(cb_w2c_in5, 1);

        // Wait for the group mask to be ready in the collector core
        noc_semaphore_wait_min(my_semaphore_ptr, 1);
        *my_semaphore_ptr = 0;

        // Let compute know we got the group masks
        cb_push_back(cb_w2c_in5, 1);

        // Wait for compute to send the top-8 values and indices
        cb_wait_front(cb_w2c_in8, 1);

        noc_async_write_one_packet_set_state</*posted=*/true>(
            collector_dst_base_addr, partials_size, /*noc=*/1, vchannel);

        // Send them over to the collector core
        noc_async_write_one_packet_with_state</*posted=*/true>(local_top8_partials_src_addr, local_top8_dst_addr);

        // Signal the collector core that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(collector_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        cb_pop_front(cb_w2c_in8, 1);

        // We should also send the raw scores
        noc_async_write_one_packet_set_state</*posted=*/true>(
            collector_dst_base_addr, raw_scores_size, /*noc=*/1, vchannel);

        // Send them over to the collector core
        noc_async_write_one_packet_with_state</*posted=*/true>(local_raw_scores_src_addr, local_raw_scores_dst_addr);

        // Signal the collector core that data is ready (increment their semaphore)
        noc_semaphore_inc</*posted=*/true>(raw_scores_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

        // Ensure write and semaphore have left the core before continuing
        noc_async_posted_atomic_barrier();

        // // Wait for combined raw scores to be ready from collector core
        // noc_semaphore_wait_min(my_raw_scores_semaphore_ptr, 1);
        // *my_raw_scores_semaphore_ptr = 0;
    }

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    if (core_id == COLLECTOR_CORE_ID) {
        // Rejig the group scores to be in the correct location -> where everyone else also puts it
        auto src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_score_src_addr1);
        auto dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_scores_dst_addr1);
        for (uint32_t i = 0; i < 8; i++) {
            dst_ptr[i] = src_ptr[i];
        }

        // Same, but for the second row of the group scores
        src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_score_src_addr2);
        dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_group_scores_dst_addr2);
        for (uint32_t i = 0; i < 8; i++) {
            dst_ptr[i] = src_ptr[i];
        }

        // We are done with the group scores
        cb_pop_front(cb_c2w_rdy, 1);

        cb_reserve_back(cb_w2c_in4, 1);

        // I am collecting, let us wait for everyone else to finish sending their data to me
        noc_semaphore_wait_min(my_semaphore_ptr, 7);
        *my_semaphore_ptr = 0;

        // Let compute know that we got the group scores
        cb_push_back(cb_w2c_in4, 1);

        //-----------------------------------------------------------------
        // Get the group masks
        //-----------------------------------------------------------------
        cb_wait_front(cb_w2c_in5, 1);
        // Multicast this data to all the cores
        noc_async_write_multicast_one_packet(
            local_group_masks_addr, group_masks_noc_addr, /*size=*/2048, /*num_dests=*/7);

        // Set the semaphore to let the clients know they got the data
        *my_semaphore_ptr = 1;
        noc_semaphore_set_multicast(
            semaphore_addr, group_semaphore_noc_addr, /*num_dests=*/7, /*linked=*/false, /*noc=*/1);

        cb_pop_front(cb_w2c_in5, 1);

        cb_reserve_back(cb_w2c_in6, 4);

        // Wait for the top8 partials to arrive
        noc_semaphore_wait_min(my_semaphore_ptr, 1 + 7);

        // Let compute know that we got the top8 partials
        cb_push_back(cb_w2c_in6, 4);

        // Wait for final top8 to be ready from compute
        cb_wait_front(cb_c2w_rdy, 1);
        cb_pop_front(cb_c2w_rdy, 1);

        // We have the top8 indices for each of 32 tokens.
        // We need to get the corresponding scores at those indices.
        // Wait for the raw scores to arrive
        noc_semaphore_wait_min(my_raw_scores_semaphore_ptr, 7);
        *my_raw_scores_semaphore_ptr = 0;

        // noc_async_write_multicast_one_packet(
        //     local_raw_scores_dst_base_addr, raw_scores_noc_addr, /*size=*/2048 * 4, /*num_dests=*/7);

        // // Set the semaphore to let the clients know they got the data
        // *my_semaphore_ptr = 1;
        // noc_semaphore_set_multicast(
        //     semaphore_addr, raw_scores_semaphore_noc_addr, /*num_dests=*/7, /*linked=*/false, /*noc=*/1);

        // Step 1: Gather the raw scores based on the top8 indices
        //
        // cb_s2c_out tile (Float16_b, 2048 bytes, no header):
        //   Face 0 (uint16 offset 0):   rows 0-7 values, rows 8-15 indices, tokens 0-15
        //   Face 1 (uint16 offset 256): rows 0-7 values, rows 8-15 indices, tokens 16-31
        //
        // cb_w2c_in7 (8 × 2048 bytes):
        //   Tile i at uint16 offset i*1024: raw scores for experts [32i..32i+31]
        //     Face 0 (+0):   experts 0-15,  tokens 0-15
        //     Face 1 (+256): experts 0-15,  tokens 16-31
        //     Face 2 (+512): experts 16-31, tokens 0-15
        //     Face 3 (+768): experts 16-31, tokens 16-31

        // Key insight from Baby RISC-V spec (RV32IM):
        //   - Max load/store width = 32 bits
        //   - L1 store throughput = 1 per 5 cycles  ← THE BOTTLENECK
        //   - 16-bit and 32-bit L1 ops cost the same → always use 32-bit
        //
        // Strategy: Process 2 adjacent tokens per iteration.
        //   - 1× lw reads 2 indices  (halves index loads: 128 vs 256)
        //   - 2× lhu reads 2 scores  (random access, can't batch)
        //   - 1× sw writes 2 scores  (halves stores:     128 vs 256)
        //
        // Cycle estimate: 128 stores × 5 cycles = 640 cycles (2× speedup)

        // auto s2c32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_src_addr);
        // auto w2c   = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_raw_scores_dst_base_addr);

        // for (uint32_t face = 0; face < 2; face++) {
        //     const uint32_t face_off   = face << 8;   // uint16 offset per face
        //     const uint32_t face_off32 = face << 7;   // uint32 offset per face

        //     for (uint32_t rank = 0; rank < 1; rank++) {
        //         const uint32_t idx_base32 = face_off32 + ((8 + rank) << 3);
        //         const uint32_t val_base32 = face_off32 + (rank << 3);

        //         for (uint32_t tp = 0; tp < 8; tp++) {
        //             // ---- 1 × 32-bit L1 load: read 2 indices ----
        //             uint32_t idx_pair = s2c32[idx_base32 + tp];
        //             uint32_t idx0 = idx_pair & 0xFFFF;
        //             uint32_t idx1 = idx_pair >> 16;

        //             uint32_t tok = tp << 1;

        //             // Decode index 0
        //             uint32_t tid0 = idx0 >> 5;
        //             uint32_t el0  = idx0 & 0x1F;
        //             uint32_t rf0  = ((el0 >> 4) << 1) | face;
        //             uint32_t rr0  = el0 & 0xF;
        //             uint32_t off0 = (tid0 << 10) | (rf0 << 8) | (rr0 << 4) | tok;

        //             // // Decode index 1
        //             // uint32_t tid1 = idx1 >> 5;
        //             // uint32_t el1  = idx1 & 0x1F;
        //             // uint32_t rf1  = ((el1 >> 4) << 1) | face;
        //             // uint32_t rr1  = el1 & 0xF;
        //             // uint32_t off1 = (tid1 << 10) | (rf1 << 8) | (rr1 << 4) | (tok + 1);

        //             // // ---- 2 × 16-bit L1 loads: read 2 scores (random, can't batch) ----
        //             // uint32_t score0 = w2c[off0];
        //             // uint32_t score1 = w2c[off1];

        //             // // ---- 1 × 32-bit L1 store: write 2 scores packed ----
        //             // s2c32[val_base32 + tp] = score0 | (score1 << 16);
        //         }
        //     }
        // }
    }
}

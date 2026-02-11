// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"

#endif

namespace deepseek_b1_ops {

// ============================================================================
// RMSNorm micro-op
//
// Computes: Update existing KV Cache with 1x576 new cache
// assumes one core with 1x512 NOPE cache and 2 cores each with 1x32 ROPE cache
//
// ============================================================================
struct KVCacheUpdate {
    // ========================================================================
    // Compile-time args structs - only what MUST be compile-time
    // (used as template parameters or in constexpr expressions)
    // ========================================================================

    // Reader CTArgs:none needed
    struct ReaderCTArgs {};

    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: none needed
    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================
    // Reader args (NCRISC): none needed
    struct ReaderArgs {
        uint32_t kv_cache_buffer_base_addr;
        uint32_t position_id;
    };
    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {
        uint32_t kv_cache_buffer_base_addr;
        uint32_t position_id;
    };
    struct ComputeArgs {
        uint32_t kv_cache_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, IsActiveCore
    // ========================================================================
    template <bool IsNopeCore, bool IsRopeCore>
    class Op {
    public:
        void operator()([[maybe_unused]] const RTArgs& args) { impl(args); }

    private:
        void dump_buffer(uint32_t readback_addr, uint32_t dump_bytes) {
            volatile tt_l1_ptr uint8_t* raw = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(readback_addr);
            for (uint32_t offset = 0; offset < dump_bytes; offset += 16) {
                DPRINT << HEX() << SETW(4) << offset << ": ";
                for (uint32_t i = 0; i < 16 && (offset + i) < dump_bytes; i++) {
                    DPRINT << SETW(2) << (uint32_t)raw[offset + i] << " ";
                }
                DPRINT << DEC() << ENDL();
            }
        }
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
            constexpr uint32_t kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb");
            constexpr uint32_t kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb");
            constexpr uint32_t PAGE_SIZE = 1088;
            constexpr uint32_t PAGES_PER_BLOCK = 18;
            constexpr uint32_t CACHES_PER_BLOCK = 32;
            constexpr uint32_t nope_num_pages = 16;

            constexpr auto k_args = TensorAccessorArgs<0>();
            auto kv_tensor_accessor = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, PAGE_SIZE);

            // This op needs to update 18 pages starting from kv_cache_page_id_start
            // If args.position_id % CACHES_PER_BLOCK != 0, there is an offset into the rows
            uint32_t kv_cache_page_id_start = args.position_id / CACHES_PER_BLOCK * PAGES_PER_BLOCK;
            uint32_t offset_in_page = args.position_id % 32;

            if constexpr (IsRopeCore) {
                constexpr uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");
                constexpr uint32_t krope_Wt = get_named_compile_time_arg_val("krope_Wt");
                constexpr uint32_t krope_num_bytes_per_core = 64;  // 1x32 float 16
                constexpr uint32_t bytes_per_datum = 2;
                constexpr uint32_t kv_cache_num_tiles = 1;

                uint32_t grid_offset_pages = 1 * (get_absolute_logical_y() - 8);
                uint32_t rope_page_id = kv_cache_page_id_start + grid_offset_pages + nope_num_pages;
                // 1. Read in data from DRAM to kv_cache_input_cb
                cb_reserve_back(kv_cache_input_cb, kv_cache_num_tiles);
                uint32_t readback_addr = get_write_ptr(kv_cache_input_cb);
                for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                    uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                    noc_async_read_page(rope_page_id, kv_tensor_accessor, cb_addr);
                }
                noc_async_read_barrier();
                dump_buffer(readback_addr, 128);
                cb_push_back(kv_cache_input_cb, kv_cache_num_tiles);

                // wait for unpacker to untilize
                cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles);

                // 2. Wait for new cache data and update into kv_cache_intermed_cb
                cb_wait_front(krope_output_cb, 1);
                // valid new rope cache from krope_output_cb
                // calculate offset in tile
                uint32_t write_addr = get_read_ptr(kv_cache_intermed_cb) + offset_in_page * krope_num_bytes_per_core;
                uint32_t new_rope_cache_addr = get_read_ptr(krope_output_cb);
                tile_info_t write_tile_info = get_tile_info(kv_cache_intermed_cb, TSLICE_INPUT_CB, TSLICE_RD_PTR);
                // Local copy: 64 bytes (1..32 bfloat16) from new_rope_cache to intermed.
                // Untilized tile layout uses a stride: first 32 bytes at write_addr, next 32 at write_addr+64.
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 1, .h1 = 2, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 2, .h1 = 3, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 3, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 4, .h1 = 5, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 5, .h1 = 6, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 6, .h1 = 7, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 7, .h1 = 8, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 8, .h1 = 9, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 9, .h1 = 10, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 10, .h1 = 11, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 11, .h1 = 12, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 12, .h1 = 13, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 13, .h1 = 14, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 14, .h1 = 15, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(
                              kv_cache_intermed_cb,
                              0,
                              SliceRange{.h0 = 15, .h1 = 16, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              TSLICE_INPUT_CB,
                              TSLICE_RD_PTR)
                       << ENDL();
                {
                    constexpr uint32_t words_per_core = (krope_num_bytes_per_core >> 2);  // 8 uint32_t per 32 bytes
                    volatile tt_l1_ptr uint32_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(new_rope_cache_addr);
                    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
                    for (uint32_t i = 0; i < words_per_core; ++i) {
                        dst[i] = src[i];
                    }
                }

                cb_pop_front(krope_output_cb, 1);
                cb_push_back(kv_cache_intermed_cb, 1);

                // 3. Wait for TRISC to finish tilize into kv_cache_output_cb and write out to DRAM
                cb_wait_front(kv_cache_output_cb, kv_cache_num_tiles);
                dump_buffer(get_read_ptr(kv_cache_output_cb), 128);
                noc_async_write_page(rope_page_id, kv_tensor_accessor, get_read_ptr(kv_cache_output_cb));
                noc_async_write_barrier();
                cb_pop_front(kv_cache_output_cb, kv_cache_num_tiles);
            }
            if constexpr (IsNopeCore) {
                constexpr uint32_t kv_cache_num_tiles = 16;

                // Wait for TRISC to finish tilize into output CB before reading (avoids race)
                cb_wait_front(kv_cache_output_cb, kv_cache_num_tiles);
                dump_buffer(get_read_ptr(kv_cache_output_cb), 256);
            }
#elif defined(COMPILE_FOR_NCRISC)
            if constexpr (IsNopeCore) {
                constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
                constexpr uint32_t kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb");
                constexpr uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
                constexpr uint32_t kv_cache_num_tiles = 16;
                constexpr uint32_t TILE_SIZE = 1088;
                constexpr auto k_args = TensorAccessorArgs<0>();
                auto kv_tensor_accessor = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, TILE_SIZE);
                cb_reserve_back(kv_cache_input_cb, kv_cache_num_tiles);
                uint32_t readback_addr = get_write_ptr(kv_cache_input_cb);
                for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                    uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                    noc_async_read_page(i, kv_tensor_accessor, cb_addr);
                }
                noc_async_read_barrier();
                dump_buffer(readback_addr, 256);
                cb_push_back(kv_cache_input_cb, kv_cache_num_tiles);
            }
#elif defined(COMPILE_FOR_TRISC)
            if constexpr (IsRopeCore) {
                constexpr uint32_t kv_cache_num_tiles = 1;
                constexpr uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");
                constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
                constexpr uint32_t kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb");
                constexpr uint32_t kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb");
                // One full 32x32 bfloat8 tile: block_ct_dim=1, full_ct_dim=1
                constexpr uint32_t full_ct_dim = 1;
                constexpr uint32_t block_ct_dim = 1;
                compute_kernel_hw_startup(kv_cache_input_cb, kv_cache_output_cb, kv_cache_intermed_cb);
                cb_wait_front(kv_cache_input_cb, kv_cache_num_tiles);
                cb_reserve_back(
                    kv_cache_intermed_cb, kv_cache_num_tiles + 1);  // one extra for ncrisc to fill in new data

                pack_untilize_init<block_ct_dim, full_ct_dim>(kv_cache_input_cb, kv_cache_intermed_cb);
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 0);
                pack_untilize_uninit(kv_cache_intermed_cb);
                cb_pop_front(kv_cache_input_cb, kv_cache_num_tiles);
                cb_push_back(kv_cache_intermed_cb, kv_cache_num_tiles);

                cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles + 1);
                cb_reserve_back(kv_cache_output_cb, kv_cache_num_tiles);

                UNPACK(reconfig_data_format_srca(kv_cache_intermed_cb));
                PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE, true>(kv_cache_output_cb)));
                tilize_init(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                tilize_block(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                // tilize_uninit(kv_cache_intermed_cb, kv_cache_output_cb);
                cb_push_back(kv_cache_output_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_intermed_cb, kv_cache_num_tiles);
            }
            if constexpr (IsNopeCore) {
                constexpr uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
                constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
                constexpr uint32_t kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb");
                constexpr uint32_t kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb");
                uint32_t kv_cache_num_tiles = args.kv_cache_num_tiles;
                constexpr uint32_t full_ct_dim = 16;
                constexpr uint32_t block_ct_dim = 8;
                compute_kernel_hw_startup(kv_cache_input_cb, kv_cache_output_cb, kv_cache_intermed_cb);
                cb_wait_front(kv_cache_input_cb, kv_cache_num_tiles);
                cb_reserve_back(kv_cache_intermed_cb, kv_cache_num_tiles);

                pack_untilize_init<block_ct_dim, full_ct_dim>(kv_cache_input_cb, kv_cache_intermed_cb);
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 0);
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 1);
                pack_untilize_uninit(kv_cache_intermed_cb);

                cb_wait_front(kv_rmsnorm_output_cb, 1);
                // valid new cache from kv_rmsnorm_output_cb
                cb_pop_front(kv_rmsnorm_output_cb, 1);

                cb_push_back(kv_cache_intermed_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_input_cb, kv_cache_num_tiles);

                cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles);
                cb_reserve_back(kv_cache_output_cb, kv_cache_num_tiles);

                UNPACK(reconfig_data_format_srca(kv_cache_intermed_cb));
                PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE, true>(kv_cache_output_cb)));
                // tilize_init(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                tilize_block(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                // tilize_uninit(kv_cache_intermed_cb, kv_cache_output_cb);
                cb_push_back(kv_cache_output_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_intermed_cb, kv_cache_num_tiles);
            }
#endif
        }
    };  // class Op

};  // struct KVCacheUpdate

}  // namespace deepseek_b1_ops

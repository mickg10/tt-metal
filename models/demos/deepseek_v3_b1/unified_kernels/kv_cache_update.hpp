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
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
            constexpr uint32_t kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb");
            constexpr uint32_t TILE_SIZE = 1088;
            constexpr uint32_t ENTRY_SIZE = 1088;
            constexpr uint32_t CHUNK_SIZE = 256;

            constexpr auto k_args = TensorAccessorArgs<0>();
            auto k_writer = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, TILE_SIZE);

            uint32_t shard_id = args.position_id / CHUNK_SIZE;
            uint32_t position_in_shard = args.position_id % CHUNK_SIZE;
            uint32_t offset_in_shard = position_in_shard * ENTRY_SIZE;

            if constexpr (IsNopeCore) {
                constexpr uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
                constexpr uint32_t kv_cache_num_tiles = 16;
                DPRINT << "wait kv_cache_intermed_cb: " << DEC() << kv_cache_intermed_cb << " " << 15 << ENDL();
                cb_wait_front(kv_cache_intermed_cb, 15);
                DPRINT << TSLICE(kv_cache_intermed_cb, 0, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 1, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 2, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 3, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 4, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 5, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 6, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 7, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 8, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 9, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 10, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 11, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 12, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 13, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 14, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 15, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                DPRINT << TSLICE(kv_cache_intermed_cb, 16, SliceRange::h0_w0_32(), TSLICE_INPUT_CB, TSLICE_RD_PTR)
                       << ENDL();
                cb_wait_front(kv_rmsnorm_output_cb, 1);
                cb_pop_front(kv_rmsnorm_output_cb, 1);
                cb_push_back(kv_cache_intermed_cb, 1);

                // Wait for TRISC to finish tilize into output CB before reading (avoids race)
                cb_wait_front(kv_cache_output_cb, kv_cache_num_tiles);
                {
                    uint32_t readback_addr = get_read_ptr(kv_cache_output_cb);
                    constexpr uint32_t dump_bytes = 656;
                    volatile tt_l1_ptr uint8_t* raw = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(readback_addr);
                    DPRINT << "kv_cache_output_cb tile0 raw bytes addr=" << HEX() << readback_addr << DEC() << ENDL();
                    for (uint32_t offset = 0; offset < dump_bytes; offset += 16) {
                        DPRINT << HEX() << SETW(4) << offset << ": ";
                        for (uint32_t i = 0; i < 16 && (offset + i) < dump_bytes; i++) {
                            DPRINT << SETW(2) << (uint32_t)raw[offset + i] << " ";
                        }
                        DPRINT << DEC() << ENDL();
                    }
                }
                uint32_t l1_read_addr = get_read_ptr(kv_cache_output_cb);
                (void)l1_read_addr;
                (void)shard_id;
            }
#elif defined(COMPILE_FOR_NCRISC)
            /*if constexpr (IsRopeCore) {
                constexpr uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");
                constexpr uint32_t krope_Wt = get_named_compile_time_arg_val("krope_Wt");
                cb_wait_front(krope_output_cb, krope_Wt);
                cb_pop_front(krope_output_cb, krope_Wt);
            }*/
            if constexpr (IsNopeCore) {
                constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
                constexpr uint32_t kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb");
                constexpr uint32_t kv_cache_num_tiles = 16;
                constexpr uint32_t TILE_SIZE = 1088;
                constexpr auto k_args = TensorAccessorArgs<0>();
                auto kv_tensor_accessor = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, TILE_SIZE);
                cb_reserve_back(kv_cache_input_cb, kv_cache_num_tiles);
                uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                uint32_t readback_addr = cb_addr;
                for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                    noc_async_read_page(i, kv_tensor_accessor, cb_addr);
                    cb_addr += TILE_SIZE;
                }
                noc_async_read_barrier();
                {
                    constexpr uint32_t dump_bytes = 656;
                    volatile tt_l1_ptr uint8_t* raw = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(readback_addr);
                    DPRINT << "kv_cache_input_cb tile0 raw bytes addr=" << HEX() << readback_addr << DEC() << ENDL();
                    for (uint32_t offset = 0; offset < dump_bytes; offset += 16) {
                        DPRINT << HEX() << SETW(4) << offset << ": ";
                        for (uint32_t i = 0; i < 16 && (offset + i) < dump_bytes; i++) {
                            DPRINT << SETW(2) << (uint32_t)raw[offset + i] << " ";
                        }
                        DPRINT << DEC() << ENDL();
                    }
                }
                cb_push_back(kv_cache_input_cb, kv_cache_num_tiles);
            }
#elif defined(COMPILE_FOR_TRISC)
            if constexpr (IsNopeCore) {
                constexpr uint32_t kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb");
                constexpr uint32_t kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb");
                constexpr uint32_t kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb");
                uint32_t kv_cache_num_tiles = args.kv_cache_num_tiles;
                constexpr uint32_t full_ct_dim = 16;
                constexpr uint32_t block_ct_dim = 8;
                compute_kernel_hw_startup(kv_cache_input_cb, kv_cache_output_cb, kv_cache_intermed_cb);
                DPRINT << "wait kv_cache_input_cb: " << DEC() << kv_cache_input_cb << " " << kv_cache_num_tiles
                       << ENDL();
                cb_wait_front(kv_cache_input_cb, kv_cache_num_tiles);
                DPRINT << "reserve kv_cache_intermed_cb: " << DEC() << kv_cache_intermed_cb << " " << kv_cache_num_tiles
                       << ENDL();
                cb_reserve_back(kv_cache_intermed_cb, kv_cache_num_tiles);

                pack_untilize_init<block_ct_dim, full_ct_dim>(kv_cache_input_cb, kv_cache_intermed_cb);
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 0);
                pack_untilize_uninit(kv_cache_intermed_cb);
                cb_push_back(kv_cache_intermed_cb, kv_cache_num_tiles - 1);
                cb_pop_front(kv_cache_input_cb, kv_cache_num_tiles);

                DPRINT << "wait kv_cache_intermed_cb: " << DEC() << kv_cache_intermed_cb << " " << kv_cache_num_tiles
                       << ENDL();
                cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles);
                cb_reserve_back(kv_cache_output_cb, kv_cache_num_tiles);

                // tilize_init(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                // tilize_block(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);

                cb_push_back(kv_cache_output_cb, kv_cache_num_tiles);
                DPRINT << "push kv_cache_output_cb: " << DEC() << kv_cache_output_cb << " " << kv_cache_num_tiles
                       << ENDL();
                cb_pop_front(kv_cache_intermed_cb, kv_cache_num_tiles);
            }
#endif
        }
    };  // class Op

};  // struct KVCacheUpdate

}  // namespace deepseek_b1_ops

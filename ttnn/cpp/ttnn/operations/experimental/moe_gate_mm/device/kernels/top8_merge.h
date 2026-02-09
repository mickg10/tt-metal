// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "compute_kernel_api/common_globals.h"

namespace ckernel {

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "lltt.h"
#include "sfpi.h"

namespace sfpu {

inline void _top8_merge_configure_addrmod_() {
    // TODO: No idea why we need this offset only when programming, but it works.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .dest = {.incr = -22, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_0);

    addr_mod_t{
        .dest = {.incr = 2, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_1);

    addr_mod_t{
        .dest = {.incr = 14, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_2);

    addr_mod_t{
        .dest = {.incr = -14, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_3);
}

template <uint32_t tile_index, uint32_t face_offset, uint32_t row_offset>
inline void _top8_merge_set_d_rwc_() {
    constexpr uint32_t total_offset = tile_index * 64 + face_offset * 32 + row_offset;

    if constexpr (total_offset < 16) {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, total_offset, 0, 0, p_setrwc::SET_D);
        return;
    }

    constexpr uint32_t num_incr = (total_offset / 8) - 1;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 8 + row_offset, 0, 0, p_setrwc::SET_D);

    for (uint32_t i = 0; i < num_incr; ++i) {
        TTI_INCRWC(p_setrwc::CR_D, 8, 0, 0);
    }
}

// Load sorted group from tile layout
template <uint32_t tile_index, uint32_t face_offset, bool replay>
inline void _top8_merge_load_rows_0_3_() {
    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 0>();

    if constexpr (replay) {
        lltt::replay(0, 4);
        _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 0>();
        lltt::replay(4, 4);
        return;
    }

    lltt::record<lltt::Exec>(0, 4);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18

    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 0>();

    lltt::record<lltt::Exec>(4, 4);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 8);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 8);  // offset 18
}

template <uint32_t tile_index, uint32_t face_offset, bool replay>
inline void _top8_merge_load_rows_4_7_() {
    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 4>();

    if constexpr (replay) {
        lltt::replay(8, 4);
        _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 4>();
        lltt::replay(12, 4);
        return;
    }

    lltt::record<lltt::Exec>(8, 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 22

    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 4>();
    lltt::record<lltt::Exec>(12, 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 8);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 20
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 8);  // offset 22
}

template <uint32_t tile_index, uint32_t face_offset, bool replay>
inline void _top8_merge_store_rows_0_3_() {
    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 0>();

    if constexpr (replay) {
        lltt::replay(16, 4);
        _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 0>();
        lltt::replay(20, 4);
        return;
    }

    lltt::record<lltt::Exec>(16, 4);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 18

    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 0>();

    lltt::record<lltt::Exec>(20, 4);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 0
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 8);  // offset 2
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 16
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 8);  // offset 18
}

template <uint32_t tile_index, uint32_t face_offset, bool replay>
inline void _top8_merge_store_rows_4_7_() {
    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 4>();

    if constexpr (replay) {
        lltt::replay(24, 4);
        _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 4>();
        lltt::replay(28, 4);
        return;
    }

    lltt::record<lltt::Exec>(24, 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);  // offset 22

    _top8_merge_set_d_rwc_<tile_index, face_offset, /*row_offset*/ 4>();

    lltt::record<lltt::Exec>(28, 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 4
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_2, 8);  // offset 6
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_1, 8);  // offset 20
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_3, 8);  // offset 22
}

// Replay Buffer Layout (32 slots total):
// Slots 0-7:   Load rows 0-3 (8 SFPLOAD instructions)
// Slots 8-15:  Load rows 4-7 (8 SFPLOAD instructions)
// Slots 16-23: Store rows 0-3 (8 SFPSTORE instructions)
// Slots 24-31: Store rows 4-7 (8 SFPSTORE instructions)
//
// Replay strategy (Option 2 - Load/Store Microkernels):
// - First call to _top8_merge_two_sorted_8_ records SFPLOAD/SFPSTORE patterns (replay=false)
// - Subsequent 6 calls replay recorded load/store patterns (replay=true)
// - Positioning (SETRWC/INCRWC) is executed separately before each replay
// - Expected speedup: ~2x on load/store (96 instructions × 6 replays = 576 inst → 288 cycles)

inline void _top8_merge_bitonic_merge_8_rows_() {
    // Stage 1: Compare distance 4 (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Stage 2: Compare distance 2 (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Stage 3: Compare distance 1 (4 swaps)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
}

// Merge two sorted-8 sequences into top-8
// Assumes LREG0-7 contains sorted sequence A (descending, untransposed)
// Loads sequence B from specified location and merges
// Result is left in LREG0-7 (untransposed, sorted descending)
template <uint32_t tile_index, uint32_t face_offset, bool replay>
inline void _top8_merge_two_sorted_8_() {
    //-------------------------------------------------------------------------
    // PHASE 1: Compare and store each halves
    //-------------------------------------------------------------------------

    // First comparison half: A[0-3] vs B[7,6,5,4]
    // Load A[0-3] to LREG0-3 (positioning + replay load pattern)
    _top8_merge_load_rows_0_3_</*tile_index*/ 0, /*face_offset*/ 0, replay>();

    // Load B[4-7] to LREG4-7 (positioning + replay load pattern)
    _top8_merge_load_rows_4_7_<tile_index, face_offset, replay>();

    // Transpose for comparison
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Perform max(A[0-3], B[7,6,5,4]) - result in LREG0-3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);

    // Transpose back for store
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store first 4 winners to temp (positioning + replay store pattern)
    _top8_merge_store_rows_0_3_</*tile_index*/ 0, /*face_offset*/ 0, replay>();

    // Second comparison half: A[4-7] vs B[3,2,1,0]
    // Load A[4-7] from temp storage (positioning + replay load pattern)
    _top8_merge_load_rows_4_7_</*tile_index*/ 0, /*face_offset*/ 0, replay>();

    // Load B[0-3] to LREG0-3 (positioning + replay load pattern)
    _top8_merge_load_rows_0_3_<tile_index, face_offset, replay>();

    // Transpose for comparison
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Perform max(A[4-7], B[3,2,1,0]) - result in LREG4-7
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);

    // Transpose back for store
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store second 4 winners to temp (positioning + replay store pattern)
    _top8_merge_store_rows_4_7_</*tile_index*/ 0, /*face_offset*/ 0, replay>();

    //-------------------------------------------------------------------------
    // PHASE 2: Bitonic merge
    //-------------------------------------------------------------------------

    // Load all 8 winners (positioning + replay load patterns)
    _top8_merge_load_rows_0_3_</*tile_index*/ 0, /*face_offset*/ 0, replay>();
    _top8_merge_load_rows_4_7_</*tile_index*/ 0, /*face_offset*/ 0, replay>();

    // Transpose for bitonic merge
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Perform bitonic merge (12 swaps) - data is transposed
    _top8_merge_bitonic_merge_8_rows_();

    // Transpose back - data is now untransposed and sorted descending
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (positioning + replay store patterns)
    _top8_merge_store_rows_0_3_</*tile_index*/ 0, /*face_offset*/ 0, replay>();
    _top8_merge_store_rows_4_7_</*tile_index*/ 0, /*face_offset*/ 0, replay>();
}

// Main entry point for cross-core merge
template <uint32_t column_idx>
inline void _top8_merge_() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Sequentially merge core 0 data with that in cores 1-7
    // First call records instructions into replay buffer (replay=false)
    // Subsequent calls replay recorded instructions (replay=true)

    // Core 1: tile 1 - Record instructions
    _top8_merge_two_sorted_8_<1, 0, /*replay=*/false>();

    // Core 2: tile 1 - Replay instructions
    _top8_merge_two_sorted_8_<1, 1, /*replay=*/true>();

    // Core 3: tile 2 - Replay instructions
    _top8_merge_two_sorted_8_<2, 0, /*replay=*/true>();

    // Core 4: tile 2 - Replay instructions
    _top8_merge_two_sorted_8_<2, 1, /*replay=*/true>();

    // Core 5: tile 3 - Replay instructions
    _top8_merge_two_sorted_8_<3, 0, /*replay=*/true>();

    // Core 6: tile 3 - Replay instructions
    _top8_merge_two_sorted_8_<3, 1, /*replay=*/true>();

    // Core 7: tile 4 - Replay instructions
    _top8_merge_two_sorted_8_<4, 0, /*replay=*/true>();

    //-------------------------------------------------------------------------
    // Create a 32-bit value for each token (lane) for each expert in the column
    //-------------------------------------------------------------------------
    _top8_merge_set_d_rwc_<0, 0, 0>();

    // Load just the top 4 indices
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_1, 8);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_2, 8);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16, ADDR_MOD_1, 8);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, 8);  // offset 18

    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // Initialize accumulator to 0 (Lower 16 experts)
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // Initialize accumulator to 0 (Higher 16 experts)

    // Main loop: 4 iterations, one per index slot
    for (uint32_t lreg = 0; lreg < 4; lreg++) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        // If value is less than "column_idx" in any lane, disable that lane
        TTI_SFPIADD((-int(column_idx << 5)) & 0xfff, lreg, lreg, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);

        // If value is greater than column_idx + 15 in any lane, disable that lane also
        TTI_SFPIADD((-int(16)) & 0xfff, lreg, p_sfpu::LREG7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_LT0);

        // Add 1 if lane is enabled
        TTI_SFPIADD(0x1, p_sfpu::LREG6, p_sfpu::LREG6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);

        // Shift left by the value in LREG[i]: LREG7 = (1 << LREG[i])
        // This creates a bitmask where bit LREG[i] is set
        TTI_SFPSHFT2(p_sfpu::LREG6, lreg, p_sfpu::LREG7, SFPSHFT2_MOD1_SHFT_LREG);

        // Now add this to the accumulator
        TTI_SFPOR(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);

        // Clear condition codes (re-enable all lanes)
        TTI_SFPENCC(0, 0, 0, 0);
    }

    // Main loop: 4 iterations, one per index slot

    for (uint32_t lreg = 0; lreg < 4; lreg++) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        // If value is less than "column_idx + 16" in any lane, disable that lane
        TTI_SFPIADD((-int(16)) & 0xfff, lreg, lreg, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);

        // If value is greater than column_idx + 15 in any lane, disable that lane also
        TTI_SFPIADD((-int(16)) & 0xfff, lreg, p_sfpu::LREG7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_LT0);

        // Add 1 if lane is enabled
        TTI_SFPIADD(0x1, p_sfpu::LREG6, p_sfpu::LREG6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);

        // Shift left by the value in LREG[i]: LREG7 = (1 << LREG[i])
        // This creates a bitmask where bit LREG[i] is set
        TTI_SFPSHFT2(p_sfpu::LREG6, lreg, p_sfpu::LREG7, SFPSHFT2_MOD1_SHFT_LREG);

        // Now add this to the accumulator
        TTI_SFPOR(0, p_sfpu::LREG7, p_sfpu::LREG5, 0);

        // Clear condition codes (re-enable all lanes)
        TTI_SFPENCC(0, 0, 0, 0);
    }

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Load just next 4 indices
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_1, 8);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_2, 8);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::LO16, ADDR_MOD_1, 8);  // offset 16
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, 8);  // offset 18

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Main loop: 4 iterations, one per index slot
    for (uint32_t lreg = 0; lreg < 4; lreg++) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        // If value is less than "column_idx" in any lane, disable that lane
        TTI_SFPIADD((-int(column_idx << 5)) & 0xfff, lreg, lreg, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);

        // If value is greater than column_idx + 15 in any lane, disable that lane also
        TTI_SFPIADD((-int(16)) & 0xfff, lreg, p_sfpu::LREG7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_LT0);

        // Add 1 if lane is enabled
        TTI_SFPIADD(0x1, p_sfpu::LREG6, p_sfpu::LREG6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);

        // Shift left by the value in LREG[i]: LREG7 = (1 << LREG[i])
        // This creates a bitmask where bit LREG[i] is set
        TTI_SFPSHFT2(p_sfpu::LREG6, lreg, p_sfpu::LREG7, SFPSHFT2_MOD1_SHFT_LREG);

        // Now add this to the accumulator
        TTI_SFPOR(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);

        // Clear condition codes (re-enable all lanes)
        TTI_SFPENCC(0, 0, 0, 0);
    }

    // Main loop: 4 iterations, one per index slot

    for (uint32_t lreg = 0; lreg < 4; lreg++) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        // If value is less than "column_idx + 16" in any lane, disable that lane
        TTI_SFPIADD((-int(16)) & 0xfff, lreg, lreg, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);

        // If value is greater than column_idx + 15 in any lane, disable that lane also
        TTI_SFPIADD((-int(16)) & 0xfff, lreg, p_sfpu::LREG7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_LT0);

        // Add 1 if lane is enabled
        TTI_SFPIADD(0x1, p_sfpu::LREG6, p_sfpu::LREG6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);

        // Shift left by the value in LREG[i]: LREG7 = (1 << LREG[i])
        // This creates a bitmask where bit LREG[i] is set
        TTI_SFPSHFT2(p_sfpu::LREG6, lreg, p_sfpu::LREG7, SFPSHFT2_MOD1_SHFT_LREG);

        // Now add this to the accumulator
        TTI_SFPOR(0, p_sfpu::LREG7, p_sfpu::LREG5, 0);

        // Clear condition codes (re-enable all lanes)
        TTI_SFPENCC(0, 0, 0, 0);
    }

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store result
    _top8_merge_set_d_rwc_<0, 1, 0>();

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16, ADDR_MOD_1, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16, ADDR_MOD_2, 0);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::LO16, ADDR_MOD_1, 0);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::LO16, ADDR_MOD_3, 0);
}

}  // namespace sfpu

inline void _llk_math_top8_merge_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_top8_merge_configure_addrmod_);
}

template <uint32_t column_idx>
inline void _llk_math_top8_merge_() {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_top8_merge_<column_idx>, 0, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the cross-core top-8 merge operation.
 */
inline void top8_merge_init() { MATH((_llk_math_top8_merge_init_())); }

/**
 * @brief Merges top-8 values across 8 cores for all 32 lanes.
 *
 * Algorithm:
 * - For each lane group (0-15 and 16-31):
 *   - Load core 0 data
 *   - Sequentially merge with cores 1-7 using bitonic merge
 *   - Store result in tile 0
 *
 * Input layout:
 * - Tiles 0-3: Each contains data from 2 cores
 * - Each core has top8 values and indices for 32 lanes (split across 2 faces)
 *
 * Output layout:
 * - Tile 0, face 0: Merged top8 for lanes 0-15
 * - Tile 0, face 1: Merged top8 for lanes 16-31
 */
template <uint32_t column_idx>
ALWI void top8_merge() {
    MATH((_llk_math_top8_merge_<column_idx>()));
}

}  // namespace ckernel

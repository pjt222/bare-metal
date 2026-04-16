/*
 * sparse_meta_int8.h — Host-side 2:4 sparse matrix utilities for INT8
 *                      mma.sp::ordered_metadata.m16n8k32.row.col (sm_86)
 *
 * Metadata format for mma.sp::ordered_metadata.m16n8k32.row.col (INT8):
 *
 *   K-step = 32 logical K-elements per IMMA-SP instruction.
 *   Compressed A: K/2 = 16 INT8 per logical row per k-step.
 *
 *   Thread gid = lane>>2 (0..7) covers output rows {gid, gid+8} of the 16-row
 *   WMMA tile. The 32-bit metadata register encodes the sparsity pattern.
 *
 *   Bit layout (sel=0x0):
 *     bits[3:0]   = nibble for K-group 0  (K-elements  0..3):  (idx1<<2)|idx0
 *     bits[7:4]   = nibble for K-group 1  (K-elements  4..7)
 *     bits[11:8]  = nibble for K-group 2  (K-elements  8..11)
 *     bits[15:12] = nibble for K-group 3  (K-elements 12..15)
 *     bits[19:16] = nibble for K-group 4  (K-elements 16..19)
 *     bits[23:20] = nibble for K-group 5  (K-elements 20..23)
 *     bits[27:24] = nibble for K-group 6  (K-elements 24..27)
 *     bits[31:28] = nibble for K-group 7  (K-elements 28..31)
 *
 *   Unlike FP16 (which duplicates bits[15:0] to bits[31:16]), INT8 uses
 *   the full 32-bit register — no duplication.
 *
 *   Hardware constraint: rows gid and gid+8 within the same 16-row tile MUST
 *   share the same nonzero positions per K-group (same as FP16 ordered_metadata).
 *
 * Metadata array layout: uint32_t[M/16 × K/32 × 8]
 *   Flattened index: (m_tile * (K/32) + k_step) * 8 + gid
 *   where m_tile = row/16, k_step = k_base/32, gid = (row%16)%8
 *
 * See also: ../hgemm_sparse/sparse_meta.h (FP16 variant, K-step=16)
 */

#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>

// Tile dimensions for mma.sp::ordered_metadata.m16n8k32
#define SPARSE_META_INT8_WMMA_M 16
#define SPARSE_META_INT8_WMMA_K 32   // logical K per IMMA-SP instruction

// Total number of uint32_t values in the metadata array
static inline size_t sparse_meta_count_int8(int M, int K) {
    return ((size_t)(M / SPARSE_META_INT8_WMMA_M))
         * ((size_t)(K / SPARSE_META_INT8_WMMA_K))
         * 8;
}

// -----------------------------------------------------------------------
// gen_random_sparse_2_4_int8
//
// Fill A_dense [M×K] with random 2:4 sparse INT8 values.
//
// Sparsity rule: within each group of 4 consecutive K-elements, exactly 2
// positions are nonzero. The two positions are chosen randomly per
// (m_tile, k_group, gid) triple. Rows (m_tile*16+gid) and
// (m_tile*16+gid+8) receive the SAME nonzero positions — satisfies the
// hardware constraint for mma.sp::ordered_metadata.
//
// Requires: K % 4 == 0, M % 16 == 0
// -----------------------------------------------------------------------
static void gen_random_sparse_2_4_int8(int8_t *A_dense, int M, int K,
                                        unsigned int seed) {
    srand(seed);
    memset(A_dense, 0, (size_t)M * K * sizeof(int8_t));

    int num_m_tiles = M / 16;
    int num_k_groups = K / 4;

    for (int m_tile = 0; m_tile < num_m_tiles; m_tile++) {
        for (int gid = 0; gid < 8; gid++) {
            int row_lo = m_tile * 16 + gid;
            int row_hi = m_tile * 16 + gid + 8;

            for (int g = 0; g < num_k_groups; g++) {
                int k_base = g * 4;

                // Pick 2 of {0,1,2,3} without replacement (Fisher-Yates partial)
                int perm[4] = {0, 1, 2, 3};
                for (int i = 3; i >= 2; i--) {
                    int j = rand() % (i + 1);
                    int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
                }
                int p0 = perm[2], p1 = perm[3];
                if (p0 > p1) { int t = p0; p0 = p1; p1 = t; }  // p0 < p1

                // Random nonzero INT8 values (-127..127, avoiding 0)
                auto rval8 = []() -> int8_t {
                    int8_t v;
                    do { v = (int8_t)((rand() % 255) - 127); }
                    while (v == 0);
                    return v;
                };

                A_dense[(size_t)row_lo * K + k_base + p0] = rval8();
                A_dense[(size_t)row_lo * K + k_base + p1] = rval8();
                A_dense[(size_t)row_hi * K + k_base + p0] = rval8();
                A_dense[(size_t)row_hi * K + k_base + p1] = rval8();
            }
        }
    }
}

// -----------------------------------------------------------------------
// compress_2_4_int8
//
// Compress dense A [M×K] with 2:4 sparsity to:
//   A_comp [M × K/2]           — packed nonzero values (INT8, row-major)
//   meta_array [(M/16)*(K/32)*8] — uint32_t metadata for mma.sp (8 nibbles)
//
// Requires: K % 32 == 0, M % 16 == 0
// -----------------------------------------------------------------------
static void compress_2_4_int8(
    const int8_t *A_dense,      // [M × K]   — input (with explicit zeros)
    int           M,
    int           K,
    int8_t       *A_comp,       // [M × K/2] — output compressed values
    uint32_t     *meta_array    // [(M/16)*(K/32)*8] — output metadata
) {
    int K_stored    = K / 2;
    int num_k_steps = K / SPARSE_META_INT8_WMMA_K;  // K/32
    int num_m_tiles = M / SPARSE_META_INT8_WMMA_M;  // M/16

    memset(meta_array, 0,
           (size_t)num_m_tiles * num_k_steps * 8 * sizeof(uint32_t));

    for (int m = 0; m < M; m++) {
        int m_tile   = m / SPARSE_META_INT8_WMMA_M;
        int tile_row = m % SPARSE_META_INT8_WMMA_M;
        int gid      = tile_row % 8;
        bool upper   = (tile_row >= 8);

        for (int ks = 0; ks < K; ks += SPARSE_META_INT8_WMMA_K) {
            int k_step_idx = ks / SPARSE_META_INT8_WMMA_K;
            int meta_idx   = (m_tile * num_k_steps + k_step_idx) * 8 + gid;

            uint32_t nibbles  = 0;
            int      comp_col = ks / 2;  // base column in compressed A row

            // 8 K-groups of 4 within this K=32 step
            for (int g = 0; g < 8; g++) {
                int k_base = ks + g * 4;

                int8_t v[4];
                for (int i = 0; i < 4; i++)
                    v[i] = A_dense[(size_t)m * K + k_base + i];

                // Find the two nonzero positions
                int idx0 = -1, idx1 = -1;
                for (int i = 0; i < 4; i++) {
                    if (v[i] != 0) {
                        if (idx0 < 0)      idx0 = i;
                        else if (idx1 < 0) { idx1 = i; break; }
                    }
                }
                if (idx0 < 0) idx0 = 0;
                if (idx1 < 0) idx1 = 1;

                // Store compressed values
                A_comp[(size_t)m * K_stored + comp_col + g * 2 + 0] = v[idx0];
                A_comp[(size_t)m * K_stored + comp_col + g * 2 + 1] = v[idx1];

                // Pack nibble: (idx1<<2)|idx0
                nibbles |= (((uint32_t)idx1 << 2) | (uint32_t)idx0) << (g * 4);
            }

            // Write metadata only for lower row of the gid pair
            // (upper row shares it — same constraint as FP16)
            if (!upper) {
                // INT8: full 32-bit register used, no duplication
                meta_array[meta_idx] = nibbles;
            }
        }
    }
}

// -----------------------------------------------------------------------
// cpu_sparse_gemm_int8
//
// CPU reference: C[M×N] = A_sparse[M×K] × B[K×N]
// A_dense has explicit zeros where values are pruned.
// Accumulates INT8 × INT8 → INT32.
//
// For small sizes only (O(M*N*K) cost).
// Loop order: row × k × col — cache-friendly.
// -----------------------------------------------------------------------
static void cpu_sparse_gemm_int8(
    const int8_t  *A_dense,  // [M × K] — with explicit zeros at sparse positions
    const int8_t  *B,        // [K × N]
    int32_t       *C,        // [M × N] — output
    int M, int N, int K
) {
    memset(C, 0, (size_t)M * N * sizeof(int32_t));
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int8_t a_val = A_dense[(size_t)row * K + k];
            if (a_val == 0) continue;
            const int8_t *b_row = &B[(size_t)k * N];
            int32_t      *c_row = &C[(size_t)row * N];
            for (int col = 0; col < N; col++) {
                c_row[col] += (int32_t)a_val * (int32_t)b_row[col];
            }
        }
    }
}

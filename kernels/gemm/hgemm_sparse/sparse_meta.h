/*
 * sparse_meta.h — Host-side 2:4 sparse matrix utilities for mma.sp::ordered_metadata
 *
 * Metadata format for mma.sp::ordered_metadata.m16n8k16.row.col (sm_86):
 *
 *   Thread gid = lane>>2 (0..7) covers output rows {gid, gid+8} of the 16-row
 *   WMMA tile. The 32-bit metadata register encodes the sparsity pattern for
 *   this thread's rows.
 *
 *   Bit layout (sel=0x0 — the only selector used in this kernel):
 *     bits[15:0]:  4 nibbles, one per K-group (K=16 → 4 groups of 4 K-elements)
 *                  nibble[g] = (idx1<<2)|idx0 where idx0 < idx1 are the two
 *                  nonzero positions within K-elements {g*4, g*4+1, g*4+2, g*4+3}
 *     bits[31:16]: duplicate of bits[15:0] (unused with sel=0x0; reserved)
 *
 *   Hardware constraint: rows gid and gid+8 within the same 16-row tile MUST
 *   share the same nonzero positions per K-group. The ordered_metadata variant
 *   only encodes 16 bits of pattern (4 nibbles) per thread, applying uniformly
 *   to both rows. If they differ, the kernel produces wrong results.
 *
 *   This constraint is satisfied by gen_random_sparse_2_4(), which assigns
 *   the same positions to both rows of each (m_tile, gid) pair.
 *
 * Metadata array layout: uint32_t[M/16 × K/16 × 8]
 *   Flattened index: (m_tile * (K/16) + k_step) * 8 + gid
 *   where m_tile = row / 16, k_step = col_group / 16, gid = (row%16) % 8
 *
 * VALIDATION NOTE: The fixed-pattern case (meta=0x44444444, positions {0,1})
 * is verified correct on sm_86. Arbitrary-pattern behavior requires GPU
 * validation — the metadata bit layout interpretation above is derived from
 * first principles and the ordered_metadata PTX ISA description, not
 * independently confirmed via hardware test of non-uniform patterns.
 */

#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// Tile dimensions for mma.sp::ordered_metadata.m16n8k16
#define SPARSE_META_WMMA_M 16
#define SPARSE_META_WMMA_K 16

// Total number of uint32_t values in the metadata array
static inline size_t sparse_meta_count(int M, int K) {
    return ((size_t)(M / SPARSE_META_WMMA_M))
         * ((size_t)(K / SPARSE_META_WMMA_K))
         * 8;
}

// -----------------------------------------------------------------------
// gen_random_sparse_2_4
//
// Fill A_dense [M×K] with random 2:4 sparse values.
//
// Sparsity rule: within each group of 4 consecutive K-elements, exactly 2
// positions are nonzero. The two positions are chosen randomly per
// (m_tile, k_group, gid) triple. Rows (m_tile*16 + gid) and
// (m_tile*16 + gid + 8) receive the SAME nonzero positions — this satisfies
// the hardware constraint for mma.sp::ordered_metadata.
//
// Requires: K % 4 == 0, M % 16 == 0
// -----------------------------------------------------------------------
static void gen_random_sparse_2_4(float *A_dense, int M, int K,
                                   unsigned int seed) {
    srand(seed);
    memset(A_dense, 0, (size_t)M * K * sizeof(float));

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
                // perm[2] and perm[3] are two randomly chosen positions
                int p0 = perm[2], p1 = perm[3];
                if (p0 > p1) { int t = p0; p0 = p1; p1 = t; }  // p0 < p1

                // Random nonzero values in (-1, 1), avoiding exact 0
                auto rval = []() -> float {
                    float v;
                    do { v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }
                    while (v == 0.0f);
                    return v;
                };

                A_dense[(size_t)row_lo * K + k_base + p0] = rval();
                A_dense[(size_t)row_lo * K + k_base + p1] = rval();
                A_dense[(size_t)row_hi * K + k_base + p0] = rval();
                A_dense[(size_t)row_hi * K + k_base + p1] = rval();
            }
        }
    }
}

// -----------------------------------------------------------------------
// compress_2_4_arbitrary
//
// Compress dense A [M×K] with 2:4 sparsity to:
//   A_comp [M × K/2]           — packed nonzero values (FP16 row-major)
//   meta_array [(M/16)*(K/16)*8] — uint32_t metadata for mma.sp
//
// How to identify nonzero positions:
//   Reads A_dense and finds the two non-zero elements per group of 4.
//   If the input violates 2:4 (e.g. 3 nonzero per group), only the first
//   two found are kept. If fewer than 2 are nonzero, positions {0,1} are
//   used as fallback.
//
// Metadata is written only for the lower row (tile_row < 8) of each gid
// pair — upper row shares the same metadata per hardware constraint.
//
// Requires: K % 16 == 0, M % 16 == 0
// -----------------------------------------------------------------------
static void compress_2_4_arbitrary(
    const float    *A_dense,   // [M × K]  — input
    int             M,
    int             K,
    __half         *A_comp,    // [M × K/2] — output compressed values
    uint32_t       *meta_array // [(M/16)*(K/16)*8] — output metadata
) {
    int K_stored    = K / 2;
    int num_k_steps = K / SPARSE_META_WMMA_K;
    int num_m_tiles = M / SPARSE_META_WMMA_M;

    memset(meta_array, 0,
           (size_t)num_m_tiles * num_k_steps * 8 * sizeof(uint32_t));

    for (int m = 0; m < M; m++) {
        int m_tile    = m / SPARSE_META_WMMA_M;
        int tile_row  = m % SPARSE_META_WMMA_M;
        int gid       = tile_row % 8;
        bool upper    = (tile_row >= 8);

        for (int ks = 0; ks < K; ks += SPARSE_META_WMMA_K) {
            int k_step_idx = ks / SPARSE_META_WMMA_K;
            int meta_idx   = (m_tile * num_k_steps + k_step_idx) * 8 + gid;

            uint32_t nibbles  = 0;
            int      comp_col = ks / 2;  // base column in compressed A row

            // Process 4 K-groups of 4 within this K=16 step
            for (int g = 0; g < 4; g++) {
                int k_base = ks + g * 4;

                float v[4];
                for (int i = 0; i < 4; i++)
                    v[i] = A_dense[(size_t)m * K + k_base + i];

                // Find the two nonzero positions
                int idx0 = -1, idx1 = -1;
                for (int i = 0; i < 4; i++) {
                    if (v[i] != 0.0f) {
                        if (idx0 < 0)      idx0 = i;
                        else if (idx1 < 0) { idx1 = i; break; }
                    }
                }
                if (idx0 < 0) idx0 = 0;
                if (idx1 < 0) idx1 = 1;

                // Store compressed values
                A_comp[(size_t)m * K_stored + comp_col + g * 2 + 0] =
                    __float2half(v[idx0]);
                A_comp[(size_t)m * K_stored + comp_col + g * 2 + 1] =
                    __float2half(v[idx1]);

                // Pack nibble: (idx1<<2)|idx0
                nibbles |= (((uint32_t)idx1 << 2) | (uint32_t)idx0) << (g * 4);
            }

            // Write metadata only for lower row of the gid pair (upper row shares it)
            if (!upper) {
                // Duplicate to upper 16 bits for potential future sel=0x1 use
                meta_array[meta_idx] = nibbles | (nibbles << 16);
            }
            // Upper row's A_comp values were packed correctly via its own nonzero
            // positions (which match the lower row if gen_random_sparse_2_4 was used).
        }
    }
}

// -----------------------------------------------------------------------
// cpu_sparse_gemm_arbitrary
//
// CPU reference: C = A_sparse × B using the actual nonzero pattern in A_dense.
// A_dense has explicit zeros where values are pruned — just multiply everything
// and the zeros contribute 0 to the accumulator naturally.
//
// For small sizes only (O(M*N*K) cost).
// -----------------------------------------------------------------------
// Loop order: row × k × col — cache-friendly (B row and C row both contiguous).
static void cpu_sparse_gemm_arbitrary(
    const float *A_dense,  // [M × K] — with explicit zeros at sparse positions
    const float *B,        // [K × N]
    float       *C,        // [M × N] — output
    int M, int N, int K
) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            float a_val = A_dense[(size_t)row * K + k];
            if (a_val == 0.0f) continue;  // skip zeros (2:4 sparsity)
            const float *b_row = &B[(size_t)k * N];
            float       *c_row = &C[(size_t)row * N];
            for (int col = 0; col < N; col++) {
                c_row[col] += a_val * b_row[col];
            }
        }
    }
}

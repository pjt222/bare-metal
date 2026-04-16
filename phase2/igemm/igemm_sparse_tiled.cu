/*
 * igemm_sparse_tiled.cu — 16-warp 128×128 Sparse INT8 GEMM with Double-Buffered cp.async
 *
 * Applies 2:4 structured sparsity to A using PTX mma.sp on INT8 Tensor Cores (sm_86).
 * Architecture mirrors hgemm_sparse_tiled.cu (FP16) — same warp grid, same pipeline,
 * same A fragment loading — with the following INT8-specific differences:
 *
 * Key differences from FP16 sparse:
 *   WMMA_K=32 (vs 16): mma.sp.m16n8k32.row.col.s32.s8.s8.s32
 *   BK=64 (2 × WMMA_K), BK_COMP=32 INT8 per row of compressed A
 *   STRIDE_A=48 bytes (32 comp + 16 pad) — same byte count as FP16 ✓
 *   STRIDE_B=144 bytes (128 + 16 pad, 16-byte aligned for cp.async)
 *   A: ldmatrix.m8n8.x2 (same lane&15 formula as FP16)
 *   B: scalar INT8 loads (ldmatrix.trans delivers b16 pairs, wrong granularity for INT8 mma)
 *   Accumulator: int32_t (dequantized to float at epilogue via scale_a × scale_b)
 *   Metadata: full 32-bit, 8 nibbles (no upper-16 duplication, unlike FP16)
 *
 * Shared memory layout (double-buffered):
 *   smem_a[2][BM × STRIDE_A] = 2 × 128 × 48 = 12,288 bytes = 12 KB
 *   smem_b[2][BK × STRIDE_B] = 2 × 64  × 144 = 18,432 bytes = 18 KB
 *   Total: 30 KB — under 50 KB cliff for 2 blocks/SM ✓
 *
 * Fragment layout (sm_86, mma.sp.m16n8k32):
 *   gid = lane >> 2 (0..7), tid_frag = lane & 3 (0..3)
 *
 *   A (compressed [16 × 32 INT8 per K-step], 2 regs):
 *     a0 = A_comp[gid][tid_frag*4..tid_frag*4+3]   (row gid,   4 INT8 packed)
 *     a1 = A_comp[gid+8][tid_frag*4..tid_frag*4+3] (row gid+8, 4 INT8 packed)
 *     Loaded via ldmatrix.m8n8.x2 — same formula as FP16 (both have 16-byte rows)
 *
 *   B (dense [32 × 8 per K-step], col-major packing, 2 regs):
 *     b0: INT8 at B[k_step + tid_frag*4 .. +3][n_col]    (K-rows 0..15, 4 per thread)
 *     b1: INT8 at B[k_step + tid_frag*4+16 .. +19][n_col] (K-rows 16..31)
 *     n_col = gid for left sub-tile, gid+8 for right sub-tile
 *     Packed little-endian: reg = b0 | b1<<8 | b2<<16 | b3<<24
 *
 *   Accumulator (4 INT32 per mma.sp call):
 *     d0 = C[gid][tid_frag*2], d1 = C[gid][tid_frag*2+1]
 *     d2 = C[gid+8][tid_frag*2], d3 = C[gid+8][tid_frag*2+1]
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin igemm_sparse_tiled.cu
 */

#include <cuda_pipeline.h>
#include <cstdint>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 32   // INT8 sparse: m16n8k32

#define BM 128
#define BN 128
#define BK 64       // 2 × WMMA_K

// A is 2:4 compressed: K_logical/2 INT8 per row
#define BK_COMP (BK / 2)   // 32

// Padding for alignment and bank-conflict reduction
// STRIDE_A: 32 + 16 = 48 bytes (16-byte aligned for ldmatrix ✓, 0 bank conflicts for x2)
// STRIDE_B: 128 + 16 = 144 bytes (16-byte aligned for cp.async ✓, 4-row bank shift=4 ✓)
#define PAD_A    16
#define PAD_B    16
#define STRIDE_A  (BK_COMP + PAD_A)   // 48 INT8 bytes per smem_a row
#define STRIDE_B  (BN + PAD_B)        // 144 INT8 bytes per smem_b row

#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 512 threads

// 4×4 warp grid over 128×128 output tile
#define WARPS_Y 4
#define WARPS_X 4

// Each warp covers 32×32 = 2×2 WMMA tiles (16×16 each)
#define WARP_TILES_M 2
#define WARP_TILES_N 2

// cp.async: 16 bytes per call
#define CP_ASYNC_BYTES 16

// A_comp: BM × BK_COMP = 128 × 32 = 4096 bytes → 256 copies of 16
// With 512 threads: guard flat < 256 (only first 256 threads load)
// B:      BK × BN  = 64  × 128 = 8192 bytes → 512 copies of 16
// Indexed as (row, chunk): row=flat/8, chunk=flat%8, col=chunk*16
// 64 rows × 8 chunks = 512 copies — exactly 1 per thread ✓

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void igemm_sparse_tiled(
    const int8_t     * __restrict__ A_compressed,  // [M × K/2] INT8 row-major
    const int8_t     * __restrict__ B,             // [K × N]   INT8 row-major
    float            * __restrict__ C,             // [M × N]   FP32 row-major (dequantized)
    const uint32_t   * __restrict__ metadata,      // [(M/16)*(K/32)*8] per-thread meta
    int M, int N, int K,
    float scale_a, float scale_b
) {
    // Double-buffered shared memory
    __shared__ __align__(16) int8_t smem_a[2][BM * STRIDE_A];     // 2 × 6 KB = 12 KB
    __shared__ __align__(16) int8_t smem_b[2][BK * STRIDE_B];     // 2 × 9 KB = 18 KB
    // Total: 30 KB — under 50 KB cliff for 2 blocks/SM ✓

    int thread_id = threadIdx.x;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;
    int wy        = warp_id / WARPS_X;   // 0..3
    int wx        = warp_id % WARPS_X;   // 0..3

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int K_stored      = K / 2;
    int K_steps_total = K / WMMA_K;   // total K=32 steps

    int gid      = lane >> 2;   // 0..7: group ID
    int tid_frag = lane & 3;    // 0..3: thread ID within group

    // ---- Accumulators: 2×2 WMMA tiles × 2 sub-tiles (left/right) × 4 INT32 ----
    int32_t acc_left[WARP_TILES_M][WARP_TILES_N][4];
    int32_t acc_right[WARP_TILES_M][WARP_TILES_N][4];
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            acc_left[wi][wj][0]  = 0;
            acc_left[wi][wj][1]  = 0;
            acc_left[wi][wj][2]  = 0;
            acc_left[wi][wj][3]  = 0;
            acc_right[wi][wj][0] = 0;
            acc_right[wi][wj][1] = 0;
            acc_right[wi][wj][2] = 0;
            acc_right[wi][wj][3] = 0;
        }
    }

    int num_tiles = (K + BK - 1) / BK;

    // ======================================================================
    // Loading macros — cp.async from global to shared memory
    // ======================================================================

    // LOAD_A_TILE: compressed A [BM × BK_COMP INT8] → smem_a[buf]
    // 4096 bytes / 16 = 256 copies. 512 threads → guard flat < 256.
    // Each copy: row r, col c ∈ {0,16} within a BK_COMP=32-wide row.
    #define LOAD_A_TILE(buf, k_base_logical)                                         \
    {                                                                                 \
        int _k_comp   = (k_base_logical) / 2;                                        \
        int _flat     = thread_id;                                                   \
        if (_flat < (BM * BK_COMP) / CP_ASYNC_BYTES) {                              \
            int _base_elem = _flat * CP_ASYNC_BYTES;                                 \
            int _row  = _base_elem / BK_COMP;                                        \
            int _col  = _base_elem % BK_COMP;                                        \
            int _soff = _row * STRIDE_A + _col;                                      \
            int _grow = block_row + _row;                                            \
            int _gcol = _k_comp + _col;                                              \
            if (_grow < M && _gcol + CP_ASYNC_BYTES - 1 < K_stored) {               \
                __pipeline_memcpy_async(                                              \
                    &smem_a[buf][_soff],                                              \
                    &A_compressed[(size_t)_grow * K_stored + _gcol],                 \
                    CP_ASYNC_BYTES);                                                  \
            } else {                                                                  \
                for (int _b = 0; _b < CP_ASYNC_BYTES; _b++) {                       \
                    int _gc = _gcol + _b;                                            \
                    smem_a[buf][_soff + _b] = (_grow < M && _gc < K_stored)          \
                        ? A_compressed[(size_t)_grow * K_stored + _gc]               \
                        : (int8_t)0;                                                  \
                }                                                                     \
            }                                                                         \
        }                                                                             \
    }

    // LOAD_B_TILE: B [BK × BN INT8] → smem_b[buf]
    // 64 rows × 8 chunks of 16 bytes = 512 copies — exactly 1 per thread.
    // Indexed by (row, chunk): row = flat/8, col = (flat%8)*16.
    // STRIDE_B=144: smem row alignment 144 mod 16 = 0 ✓; col mod 16 = 0 ✓.
    #define LOAD_B_TILE(buf, k_base_logical)                                         \
    {                                                                                 \
        int _flat  = thread_id;                                                      \
        int _row   = _flat / (BN / CP_ASYNC_BYTES);                                 \
        int _col   = (_flat % (BN / CP_ASYNC_BYTES)) * CP_ASYNC_BYTES;              \
        int _soff  = _row * STRIDE_B + _col;                                         \
        int _grow  = (k_base_logical) + _row;                                       \
        int _gcol  = block_col + _col;                                               \
        if (_grow < K && _gcol + CP_ASYNC_BYTES - 1 < N) {                          \
            __pipeline_memcpy_async(                                                  \
                &smem_b[buf][_soff],                                                  \
                &B[(size_t)_grow * N + _gcol],                                       \
                CP_ASYNC_BYTES);                                                      \
        } else {                                                                      \
            for (int _b = 0; _b < CP_ASYNC_BYTES; _b++) {                           \
                int _gn = _gcol + _b;                                                \
                smem_b[buf][_soff + _b] = (_grow < K && _gn < N)                    \
                    ? B[(size_t)_grow * N + _gn]                                     \
                    : (int8_t)0;                                                      \
            }                                                                         \
        }                                                                             \
    }

    // ======================================================================
    // COMPUTE_TILE: process one K-tile from shared memory using mma.sp
    //
    // A fragments: ldmatrix.sync.aligned.m8n8.x2.shared.b16
    //   Same formula as FP16: lane & 15 selects one of 16 rows.
    //   Works because INT8 K_comp rows are also 16 bytes (32 INT8).
    //   fa0 covers rows 0-7 (gid=0..7), fa1 covers rows 8-15 (gid=8..15).
    //
    // B fragments: scalar INT8 loads (8 LDS.U8 → 2 registers of 4 INT8 each)
    //   b0: B[k_step + tid_frag*4 .. +3][n_col]    — K-rows 0..15 of this sub-step
    //   b1: B[k_step + tid_frag*4+16 .. +19][n_col] — K-rows 16..31 of this sub-step
    //   Packed little-endian in uint32_t.
    //
    // Metadata: full 32-bit, 8 nibbles (no duplication unlike FP16).
    // ======================================================================
    #define COMPUTE_TILE(buf, tile_idx)                                              \
    _Pragma("unroll")                                                                \
    for (int k_step = 0; k_step < BK; k_step += WMMA_K) {                          \
        int k_comp_offset = k_step / 2;   /* 0 or 16 INT8 */                        \
        int _gk_idx = (tile_idx) * (BK / WMMA_K) + k_step / WMMA_K;               \
        /* K-row offsets for B scalar loads */                                       \
        int _k0 = k_step + tid_frag * 4;         /* rows 0..15 of K-step */         \
        int _k1 = k_step + tid_frag * 4 + 16;    /* rows 16..31 of K-step */        \
                                                                                     \
        _Pragma("unroll")                                                            \
        for (int wi = 0; wi < WARP_TILES_M; wi++) {                                 \
            int a_base_row  = wy * 32 + wi * WMMA_M;                                \
            int _m_tile_idx = block_row / WMMA_M + wy * WARP_TILES_M + wi;         \
            uint32_t meta = metadata[                                                \
                (size_t)_m_tile_idx * K_steps_total * 8                             \
              + (size_t)_gk_idx     * 8                                             \
              + gid];                                                                \
                                                                                     \
            /* A fragment via ldmatrix.m8n8.x2 */                                   \
            uint32_t fa0, fa1;                                                       \
            {                                                                        \
                uint32_t _smem_a_ptr = __cvta_generic_to_shared(                    \
                    &smem_a[buf][(a_base_row + (lane & 15)) * STRIDE_A               \
                                  + k_comp_offset]);                                 \
                asm volatile(                                                        \
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"    \
                    : "=r"(fa0), "=r"(fa1)                                           \
                    : "r"(_smem_a_ptr));                                             \
            }                                                                        \
                                                                                     \
            _Pragma("unroll")                                                        \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                             \
                int b_base_col  = wx * 32 + wj * WMMA_N;                            \
                int _ncol_left  = b_base_col + gid;        /* gid=0..7 */           \
                int _ncol_right = b_base_col + 8 + gid;                             \
                                                                                     \
                /* B left sub-tile: pack 4 INT8 for b0, 4 for b1 */                 \
                uint32_t fb_left0, fb_left1;                                         \
                {                                                                    \
                    const int8_t *_p = &smem_b[buf][_k0 * STRIDE_B + _ncol_left];  \
                    fb_left0 = (uint32_t)(uint8_t)_p[0]                             \
                             | ((uint32_t)(uint8_t)_p[STRIDE_B]     << 8)           \
                             | ((uint32_t)(uint8_t)_p[STRIDE_B * 2] << 16)          \
                             | ((uint32_t)(uint8_t)_p[STRIDE_B * 3] << 24);         \
                    const int8_t *_q = &smem_b[buf][_k1 * STRIDE_B + _ncol_left];  \
                    fb_left1 = (uint32_t)(uint8_t)_q[0]                             \
                             | ((uint32_t)(uint8_t)_q[STRIDE_B]     << 8)           \
                             | ((uint32_t)(uint8_t)_q[STRIDE_B * 2] << 16)          \
                             | ((uint32_t)(uint8_t)_q[STRIDE_B * 3] << 24);         \
                }                                                                    \
                                                                                     \
                /* B right sub-tile */                                               \
                uint32_t fb_right0, fb_right1;                                       \
                {                                                                    \
                    const int8_t *_p = &smem_b[buf][_k0 * STRIDE_B + _ncol_right]; \
                    fb_right0 = (uint32_t)(uint8_t)_p[0]                            \
                              | ((uint32_t)(uint8_t)_p[STRIDE_B]     << 8)          \
                              | ((uint32_t)(uint8_t)_p[STRIDE_B * 2] << 16)         \
                              | ((uint32_t)(uint8_t)_p[STRIDE_B * 3] << 24);        \
                    const int8_t *_q = &smem_b[buf][_k1 * STRIDE_B + _ncol_right]; \
                    fb_right1 = (uint32_t)(uint8_t)_q[0]                            \
                              | ((uint32_t)(uint8_t)_q[STRIDE_B]     << 8)          \
                              | ((uint32_t)(uint8_t)_q[STRIDE_B * 2] << 16)         \
                              | ((uint32_t)(uint8_t)_q[STRIDE_B * 3] << 24);        \
                }                                                                    \
                                                                                     \
                /* mma.sp: D_left += A_sparse × B_left (INT8→INT32) */              \
                asm volatile(                                                        \
                    "mma.sp::ordered_metadata.sync.aligned"                          \
                    ".m16n8k32.row.col.s32.s8.s8.s32 "                              \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "                             \
                    "{%8,%9,%10,%11}, %12, 0x0;\n"                                   \
                    : "=r"(acc_left[wi][wj][0]), "=r"(acc_left[wi][wj][1]),          \
                      "=r"(acc_left[wi][wj][2]), "=r"(acc_left[wi][wj][3])          \
                    : "r"(fa0), "r"(fa1),                                            \
                      "r"(fb_left0), "r"(fb_left1),                                  \
                      "r"(acc_left[wi][wj][0]), "r"(acc_left[wi][wj][1]),            \
                      "r"(acc_left[wi][wj][2]), "r"(acc_left[wi][wj][3]),            \
                      "r"(meta));                                                     \
                                                                                     \
                /* mma.sp: D_right += A_sparse × B_right (INT8→INT32) */            \
                asm volatile(                                                        \
                    "mma.sp::ordered_metadata.sync.aligned"                          \
                    ".m16n8k32.row.col.s32.s8.s8.s32 "                              \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "                             \
                    "{%8,%9,%10,%11}, %12, 0x0;\n"                                   \
                    : "=r"(acc_right[wi][wj][0]), "=r"(acc_right[wi][wj][1]),        \
                      "=r"(acc_right[wi][wj][2]), "=r"(acc_right[wi][wj][3])        \
                    : "r"(fa0), "r"(fa1),                                            \
                      "r"(fb_right0), "r"(fb_right1),                                \
                      "r"(acc_right[wi][wj][0]), "r"(acc_right[wi][wj][1]),          \
                      "r"(acc_right[wi][wj][2]), "r"(acc_right[wi][wj][3]),          \
                      "r"(meta));                                                     \
            }                                                                        \
        }                                                                            \
    }

    // ======================================================================
    // Double-buffered pipeline
    // ======================================================================

    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        LOAD_A_TILE(next_buf, next_k_base);
        LOAD_B_TILE(next_buf, next_k_base);
        __pipeline_commit();
        COMPUTE_TILE(cur_buf, tile);
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    {
        int last_buf = (num_tiles - 1) & 1;
        COMPUTE_TILE(last_buf, num_tiles - 1);
    }

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // ======================================================================
    // Dequantize and store
    // Accumulator layout: d0=C[gid][tid*2], d1=C[gid][tid*2+1]
    //                     d2=C[gid+8][tid*2], d3=C[gid+8][tid*2+1]
    // ======================================================================
    float dequant_scale = scale_a * scale_b;
    int store_col0 = tid_frag * 2;
    int store_col1 = store_col0 + 1;

    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_base_row = block_row + wy * 32 + wi * WMMA_M;
            int c_base_col = block_col + wx * 32 + wj * WMMA_N;

            int row_lo = c_base_row + gid;
            int row_hi = c_base_row + gid + 8;

            // Left sub-tile (cols 0..7 within WMMA_N=16 tile)
            if (row_lo < M) {
                if (c_base_col + store_col0 < N)
                    C[(size_t)row_lo * N + c_base_col + store_col0] =
                        (float)acc_left[wi][wj][0] * dequant_scale;
                if (c_base_col + store_col1 < N)
                    C[(size_t)row_lo * N + c_base_col + store_col1] =
                        (float)acc_left[wi][wj][1] * dequant_scale;
            }
            if (row_hi < M) {
                if (c_base_col + store_col0 < N)
                    C[(size_t)row_hi * N + c_base_col + store_col0] =
                        (float)acc_left[wi][wj][2] * dequant_scale;
                if (c_base_col + store_col1 < N)
                    C[(size_t)row_hi * N + c_base_col + store_col1] =
                        (float)acc_left[wi][wj][3] * dequant_scale;
            }

            // Right sub-tile (cols 8..15 within WMMA_N=16 tile)
            if (row_lo < M) {
                if (c_base_col + 8 + store_col0 < N)
                    C[(size_t)row_lo * N + c_base_col + 8 + store_col0] =
                        (float)acc_right[wi][wj][0] * dequant_scale;
                if (c_base_col + 8 + store_col1 < N)
                    C[(size_t)row_lo * N + c_base_col + 8 + store_col1] =
                        (float)acc_right[wi][wj][1] * dequant_scale;
            }
            if (row_hi < M) {
                if (c_base_col + 8 + store_col0 < N)
                    C[(size_t)row_hi * N + c_base_col + 8 + store_col0] =
                        (float)acc_right[wi][wj][2] * dequant_scale;
                if (c_base_col + 8 + store_col1 < N)
                    C[(size_t)row_hi * N + c_base_col + 8 + store_col1] =
                        (float)acc_right[wi][wj][3] * dequant_scale;
            }
        }
    }
}

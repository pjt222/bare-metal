/*
 * hgemm_sparse_tiled.cu — 16-warp 128×128 Sparse HGEMM with Double-Buffered cp.async
 *
 * Combines the tiling/double-buffering architecture from hgemm_16warp.cu with
 * the PTX mma.sp sparse Tensor Core path from hgemm_sparse_naive.cu.
 *
 * Design:
 *   Output tile: 128×128 (BM=128, BN=128)
 *   K-tile:      BK=32 (32 logical K elements per tile; 16 compressed for A)
 *   Warps:       16 (4×4 grid), each covering 32×32 = 2×2 WMMA tiles (16×16)
 *   Block size:  512 threads
 *
 * Shared memory layout (double-buffered):
 *   smem_a[2][BM × STRIDE_A] — A is 2:4 compressed, so K dim is BK/2=16
 *     STRIDE_A = BK/2 + PAD = 16 + 8 = 24 halfs
 *     Per buffer: 128 × 24 × 2 = 6,144 bytes = 6 KB
 *   smem_b[2][BK × STRIDE_B] — B is dense
 *     STRIDE_B = BN + PAD = 128 + 8 = 136 halfs
 *     Per buffer: 32 × 136 × 2 = 8,704 bytes ≈ 8.5 KB
 *   Total double-buffered: 2 × (6 + 8.5) = ~29 KB — well under 50 KB cliff
 *
 * Compute path:
 *   Uses PTX mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *   (NOT the WMMA API — fragments are constructed manually from shared memory)
 *
 *   Per warp, per K-step (16 logical K = 8 compressed K):
 *     4 output tiles × 2 sub-tiles (left n=0..7, right n=8..15) = 8 mma.sp calls
 *   Per K-tile (BK=32 = 2 K-steps): 16 mma.sp per warp
 *   Total per tile: 16 warps × 16 = 256 HMMA.SP instructions
 *
 * Fragment layout (sm_86, empirically verified):
 *   gid = lane >> 2 (0..7), tid = lane & 3 (0..3)
 *
 *   A (sparse compressed [16 × K/2], 2 regs per m16n8k16):
 *     a0 = {A_comp[gid][tid*2],   A_comp[gid][tid*2+1]}     row gid
 *     a1 = {A_comp[gid+8][tid*2], A_comp[gid+8][tid*2+1]}   row gid+8
 *
 *   B (dense [K × N], col-major reg packing, 2 regs per m16n8k16):
 *     b0 = {B[tid*2][n],   B[tid*2+1][n]}     K rows tid*2..tid*2+1
 *     b1 = {B[tid*2+8][n], B[tid*2+9][n]}     K rows tid*2+8..tid*2+9
 *     n = gid for left sub-tile, gid+8 for right sub-tile
 *
 *   Accumulator (4 FP32 regs per m16n8k16):
 *     d0 = C[gid][tid*2],   d1 = C[gid][tid*2+1]
 *     d2 = C[gid+8][tid*2], d3 = C[gid+8][tid*2+1]
 *
 * Metadata: loaded dynamically from metadata[] array (see sparse_meta.h).
 *   Each uint32_t entry covers one gid (lane>>2, 0..7) for one 16-row M-tile
 *   and one K=16 step. bits[15:0] encode 4 nibbles (K-groups 0–3); upper 16
 *   bits duplicate the lower 16 (reserved). Rows gid and gid+8 share the same
 *   nibbles — hardware constraint of mma.sp::ordered_metadata (see sparse_meta.h).
 *
 * A fragment loading: ldmatrix.sync.aligned.m8n8.x2.shared.b16
 *   Lane t → &smem_a[(a_base_row + (t & 15)) * STRIDE_A + k_comp_offset]
 *   fa0 (rows 0–7) and fa1 (rows 8–15) in one instruction.
 *   Eliminates 4×LDS.U16 + 2×PRMT per register pair (verified on sm_86).
 *   A fragment reuse: ldmatrix is outside the wj loop, fa0/fa1 reused across
 *   both wj tiles (no redundant reload within a k_step).
 *
 * B fragment loading: ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
 *   Works with the EXISTING row-major smem_b ([BK × STRIDE_B]) — no layout change.
 *   With .trans, each thread provides a pointer to a K-row (16 contiguous bytes
 *   along N), and the hardware transposes to deliver the correct K-column values
 *   per N-column to each thread. Address per thread:
 *     row = k_step + (lane&7) + ((lane>>3)&1)*8
 *     ptr = &smem_b[row * STRIDE_B + b_base_col + {0 or 8}]
 *   Result: thread (gid=g, tid_frag=t) receives
 *     fb0 = packed{B[k_step+t*2][b_base_col+g], B[k_step+t*2+1][b_base_col+g]}
 *     fb1 = packed{B[k_step+t*2+8][b_base_col+g], B[k_step+t*2+9][b_base_col+g]}
 *   Replaces 8 LDS.U16 + 4 PRMT per wj with 2 LDSM.16.M88.2 — same 4-reg footprint.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_sparse_tiled.sm_86.cubin hgemm_sparse_tiled.cu
 */

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdint>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 32

// A is 2:4 compressed: logical K compressed to K/2
#define BK_COMP (BK / 2)   // 16

// Bank-conflict-free padding
#define PAD_A    8
#define PAD_B    8
#define STRIDE_A  (BK_COMP + PAD_A)  // 16 + 8 = 24 halfs (A row stride)
#define STRIDE_B  (BN + PAD_B)       // 128 + 8 = 136 halfs (B row stride)

#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 512 threads

// 4×4 warp grid over 128×128 output tile
#define WARPS_Y 4
#define WARPS_X 4

// Each warp covers 32×32 = 2×2 WMMA tiles (16×16 each)
#define WARP_TILES_M 2
#define WARP_TILES_N 2

// cp.async: 16 bytes per call = 8 __half values
#define CP_ASYNC_BYTES 16
#define ELEMS_PER_COPY (CP_ASYNC_BYTES / 2)  // 8 halfs

// A_compressed tile: BM × BK_COMP = 128 × 16 = 2048 halfs = 4096 bytes
// 4096 / 16 = 256 copies needed. With 512 threads, ~0.5 per thread → 1 iter with half threads active
// B tile: BK × BN = 32 × 128 = 4096 halfs = 8192 bytes
// 8192 / 16 = 512 copies needed. 512 threads → 1 iter (exactly one per thread)

// Number of cp.async iterations per thread
#define CP_ITERS_A 1   // 256 copies / 512 threads → 0.5, guard with flat < total
#define CP_ITERS_B 1   // 512 copies / 512 threads → 1.0, guard with flat < total

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void hgemm_sparse_tiled(
    const __half     * __restrict__ A_compressed,  // [M × K/2] FP16 row-major
    const __half     * __restrict__ B,             // [K × N]   FP16 row-major
    float            * __restrict__ C,             // [M × N]   FP32 row-major
    const uint32_t   * __restrict__ metadata,      // [(M/16)*(K/16)*8] per-thread meta
    int M, int N, int K
) {
    // Double-buffered shared memory
    __shared__ __align__(16) __half smem_a[2][BM * STRIDE_A];       // 2 × 6 KB = 12 KB
    __shared__ __align__(16) __half smem_b[2][BK * STRIDE_B];       // 2 × 8.5 KB = 17 KB
    // smem_b layout: row-major [BK × STRIDE_B]
    //   smem_b[buf][k * STRIDE_B + n] = B[k_tile_base + k][block_col + n]
    // Total: 12 + 17 = 29 KB — under 50 KB cliff for 2 blocks/SM ✓

    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane = thread_id % WARP_SIZE;
    int wy = warp_id / WARPS_X;   // 0..3 — warp's row in the 4×4 grid
    int wx = warp_id % WARPS_X;   // 0..3 — warp's column in the 4×4 grid

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int K_stored      = K / 2;         // compressed K dimension of A
    int K_steps_total = K / WMMA_K;   // total K=16 steps across full K dimension

    // Fragment indexing within a warp
    int gid      = lane >> 2;   // 0..7: group ID (selects M-row within 16×8 tile)
    int tid_frag = lane & 3;    // 0..3: thread ID within group (selects K position)

    // ---- Accumulators: 2×2 WMMA tiles × 2 sub-tiles (left/right) × 4 regs ----
    float acc_left[WARP_TILES_M][WARP_TILES_N][4];
    float acc_right[WARP_TILES_M][WARP_TILES_N][4];
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            acc_left[wi][wj][0] = 0.0f;
            acc_left[wi][wj][1] = 0.0f;
            acc_left[wi][wj][2] = 0.0f;
            acc_left[wi][wj][3] = 0.0f;
            acc_right[wi][wj][0] = 0.0f;
            acc_right[wi][wj][1] = 0.0f;
            acc_right[wi][wj][2] = 0.0f;
            acc_right[wi][wj][3] = 0.0f;
        }
    }

    int num_tiles = (K + BK - 1) / BK;

    // ======================================================================
    // Loading macros — cp.async from global to shared memory
    // ======================================================================

    // LOAD_A_TILE: load compressed A tile [BM × BK_COMP] into smem_a[buf]
    // Total elements: 128 × 16 = 2048 halfs. Each cp.async copies 8 halfs.
    // 2048 / 8 = 256 copies needed. 512 threads → only first 256 threads load.
    #define LOAD_A_TILE(buf, k_base_logical)                                      \
    {                                                                              \
        int _k_comp = (k_base_logical) / 2;                                       \
        int _flat = thread_id * ELEMS_PER_COPY;                                   \
        if (_flat < BM * BK_COMP) {                                               \
            int _row = _flat / BK_COMP;                                           \
            int _col = _flat % BK_COMP;                                           \
            int _soff = _row * STRIDE_A + _col;                                   \
            int _grow = block_row + _row;                                         \
            int _gcol = _k_comp + _col;                                           \
            if (_grow < M && _gcol + ELEMS_PER_COPY - 1 < K_stored) {            \
                __pipeline_memcpy_async(                                           \
                    &smem_a[buf][_soff],                                           \
                    &A_compressed[(size_t)_grow * K_stored + _gcol],              \
                    CP_ASYNC_BYTES);                                               \
            } else {                                                               \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {                    \
                    int _gc = _gcol + _b;                                         \
                    smem_a[buf][_soff + _b] = (_grow < M && _gc < K_stored)       \
                        ? A_compressed[(size_t)_grow * K_stored + _gc]            \
                        : __float2half(0.0f);                                     \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }

    // LOAD_B_TILE: load dense B[k_base..k_base+BK-1][block_col..block_col+BN-1]
    // into smem_b[buf] via cp.async (row-major layout, K×STRIDE_B).
    // Total elements: BK × BN = 32 × 128 = 4096 halfs = 8192 bytes.
    // Each cp.async copies 16 bytes = 8 halfs. 512 copies needed → 1 per thread.
    #define LOAD_B_TILE(buf, k_base_logical)                                      \
    {                                                                              \
        int _flat = thread_id * ELEMS_PER_COPY;                                   \
        if (_flat < BK * BN) {                                                    \
            int _row = _flat / BN;   /* K row within tile */                      \
            int _col = _flat % BN;   /* N col within tile */                      \
            int _soff = _row * STRIDE_B + _col;                                   \
            int _grow = (k_base_logical) + _row;                                  \
            int _gcol = block_col + _col;                                         \
            if (_grow < K && _gcol + ELEMS_PER_COPY - 1 < N) {                   \
                __pipeline_memcpy_async(                                           \
                    &smem_b[buf][_soff],                                           \
                    &B[(size_t)_grow * N + _gcol],                                \
                    CP_ASYNC_BYTES);                                               \
            } else {                                                               \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {                    \
                    int _gn = _gcol + _b;                                         \
                    smem_b[buf][_soff + _b] = (_grow < K && _gn < N)              \
                        ? B[(size_t)_grow * N + _gn]                              \
                        : __float2half(0.0f);                                     \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }

    // ======================================================================
    // COMPUTE_TILE: process one K-tile from shared memory using mma.sp
    //
    // A fragments: loaded via ldmatrix.sync.aligned.m8n8.x2.shared.b16
    //   Lane t provides smem address: &smem_a[(a_base_row + gid) * STRIDE_A + k_comp_offset]
    //   All 4 lanes sharing the same gid provide the same row address; ldmatrix
    //   distributes the 8 halfs of that row across tid_frag (cols tid*2, tid*2+1).
    //   Produces fa0 (A rows 0–7) and fa1 (A rows 8–15) in one instruction,
    //   replacing 4×LDS.U16 + 2×PRMT per register pair.
    //
    // B fragments: scalar LDS (ldmatrix.trans not applicable — smem_b is row-major,
    //   K-rows are strided by STRIDE_B halfs, not contiguous as .trans requires)
    // ======================================================================
    // COMPUTE_TILE(buf, tile_idx):
    //   buf      — double-buffer index (0 or 1)
    //   tile_idx — K-tile loop index, used to compute global k_step index for
    //              metadata lookup: global_k_step = tile_idx*(BK/WMMA_K) + local
    #define COMPUTE_TILE(buf, tile_idx)                                            \
    _Pragma("unroll")                                                              \
    for (int k_step = 0; k_step < BK; k_step += WMMA_K) {                        \
        /* k_step is the logical K offset within the tile (0 or 16) */            \
        int k_comp_offset = k_step / 2;  /* compressed K offset: 0 or 8 */       \
        /* Global K=16 step index for metadata lookup */                           \
        int _gk_idx = (tile_idx) * (BK / WMMA_K) + k_step / WMMA_K;             \
        /* B-fragment smem row for ldmatrix.trans:                               */\
        /* Thread lane t provides K-row k_step+(lane&7) for matrix-0 (K-rows    */\
        /* 0..7) and k_step+(lane&7)+8 for matrix-1 (K-rows 8..15).             */\
        int _b_k_row = k_step + (lane & 7) + (((lane >> 3) & 1) * 8);           \
        int _b_smem_k_off = _b_k_row * STRIDE_B;                                  \
                                                                                   \
        _Pragma("unroll")                                                          \
        for (int wi = 0; wi < WARP_TILES_M; wi++) {                               \
            /* A fragment via ldmatrix.sync.aligned.m8n8.x2:                    */\
            /* Lane t provides address for row (a_base_row + t%16).             */\
            /* fa0 gets rows 0-7, fa1 gets rows 8-15.                           */\
            int a_base_row = wy * 32 + wi * WMMA_M;                               \
            int _m_tile_idx = block_row / WMMA_M + wy * WARP_TILES_M + wi;       \
            uint32_t meta = metadata[                                              \
                (size_t)_m_tile_idx * K_steps_total * 8                           \
              + (size_t)_gk_idx     * 8                                           \
              + gid];                                                              \
            uint32_t fa0, fa1;                                                     \
            {                                                                      \
                uint32_t smem_a_ptr = __cvta_generic_to_shared(                   \
                    &smem_a[buf][(a_base_row + (lane & 15)) * STRIDE_A            \
                                 + k_comp_offset]);                                \
                asm volatile(                                                      \
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"  \
                    : "=r"(fa0), "=r"(fa1)                                        \
                    : "r"(smem_a_ptr));                                            \
            }                                                                      \
                                                                                   \
            _Pragma("unroll")                                                      \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                           \
                /* B fragment via ldmatrix.sync.aligned.m8n8.x2.trans:          */\
                /* Each thread points to a K-row (16 contiguous N-bytes).        */\
                /* .trans delivers the N-column for this thread's gid.           */\
                /* Left sub-tile:  N-cols b_base_col+[0..7]  (gid selects)      */\
                /* Right sub-tile: N-cols b_base_col+[8..15] (gid selects +8)   */\
                int b_base_col = wx * 32 + wj * WMMA_N;                           \
                uint32_t fb_left0, fb_left1;                                      \
                {                                                                  \
                    uint32_t _ptr = __cvta_generic_to_shared(                     \
                        &smem_b[buf][_b_smem_k_off + b_base_col]);                \
                    asm volatile(                                                  \
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"         \
                        " {%0,%1}, [%2];\n"                                       \
                        : "=r"(fb_left0), "=r"(fb_left1)                         \
                        : "r"(_ptr));                                              \
                }                                                                  \
                uint32_t fb_right0, fb_right1;                                    \
                {                                                                  \
                    uint32_t _ptr = __cvta_generic_to_shared(                     \
                        &smem_b[buf][_b_smem_k_off + b_base_col + 8]);            \
                    asm volatile(                                                  \
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"         \
                        " {%0,%1}, [%2];\n"                                       \
                        : "=r"(fb_right0), "=r"(fb_right1)                       \
                        : "r"(_ptr));                                              \
                }                                                                  \
                                                                                   \
                /* mma.sp: C_left += A_sparse × B_left */                         \
                asm volatile(                                                      \
                    "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col"      \
                    ".f32.f16.f16.f32 "                                           \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "                           \
                    "{%8,%9,%10,%11}, %12, 0x0;\n"                                \
                    : "=f"(acc_left[wi][wj][0]), "=f"(acc_left[wi][wj][1]),       \
                      "=f"(acc_left[wi][wj][2]), "=f"(acc_left[wi][wj][3])       \
                    : "r"(fa0), "r"(fa1),                                         \
                      "r"(fb_left0), "r"(fb_left1),                               \
                      "f"(acc_left[wi][wj][0]), "f"(acc_left[wi][wj][1]),         \
                      "f"(acc_left[wi][wj][2]), "f"(acc_left[wi][wj][3]),         \
                      "r"(meta));                                                  \
                                                                                   \
                /* mma.sp: C_right += A_sparse × B_right */                       \
                asm volatile(                                                      \
                    "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col"      \
                    ".f32.f16.f16.f32 "                                           \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "                           \
                    "{%8,%9,%10,%11}, %12, 0x0;\n"                                \
                    : "=f"(acc_right[wi][wj][0]), "=f"(acc_right[wi][wj][1]),     \
                      "=f"(acc_right[wi][wj][2]), "=f"(acc_right[wi][wj][3])     \
                    : "r"(fa0), "r"(fa1),                                         \
                      "r"(fb_right0), "r"(fb_right1),                             \
                      "f"(acc_right[wi][wj][0]), "f"(acc_right[wi][wj][1]),       \
                      "f"(acc_right[wi][wj][2]), "f"(acc_right[wi][wj][3]),       \
                      "r"(meta));                                                  \
            }                                                                      \
        }                                                                          \
    }

    // ======================================================================
    // Double-buffered pipeline
    // ======================================================================

    // Prologue: load first tile into buffer 0
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Main loop: overlap loading next tile with computing current tile
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf  = tile & 1;
        int next_buf = 1 - cur_buf;

        LOAD_A_TILE(next_buf, next_k_base);
        LOAD_B_TILE(next_buf, next_k_base);
        __pipeline_commit();
        COMPUTE_TILE(cur_buf, tile);
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // Epilogue: compute last tile
    {
        int last_buf = (num_tiles - 1) & 1;
        COMPUTE_TILE(last_buf, num_tiles - 1);
    }

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // ======================================================================
    // Store accumulators to global memory
    // ======================================================================
    // Accumulator layout (sm_86): d0 = C[gid][tid*2], d1 = C[gid][tid*2+1]
    //                              d2 = C[gid+8][tid*2], d3 = C[gid+8][tid*2+1]
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

            // Left sub-tile (n=0..7 within the 16×16 WMMA tile)
            if (row_lo < M) {
                if (c_base_col + store_col0 < N)
                    C[(size_t)row_lo * N + c_base_col + store_col0] = acc_left[wi][wj][0];
                if (c_base_col + store_col1 < N)
                    C[(size_t)row_lo * N + c_base_col + store_col1] = acc_left[wi][wj][1];
            }
            if (row_hi < M) {
                if (c_base_col + store_col0 < N)
                    C[(size_t)row_hi * N + c_base_col + store_col0] = acc_left[wi][wj][2];
                if (c_base_col + store_col1 < N)
                    C[(size_t)row_hi * N + c_base_col + store_col1] = acc_left[wi][wj][3];
            }

            // Right sub-tile (n=8..15 within the 16×16 WMMA tile)
            if (row_lo < M) {
                if (c_base_col + 8 + store_col0 < N)
                    C[(size_t)row_lo * N + c_base_col + 8 + store_col0] = acc_right[wi][wj][0];
                if (c_base_col + 8 + store_col1 < N)
                    C[(size_t)row_lo * N + c_base_col + 8 + store_col1] = acc_right[wi][wj][1];
            }
            if (row_hi < M) {
                if (c_base_col + 8 + store_col0 < N)
                    C[(size_t)row_hi * N + c_base_col + 8 + store_col0] = acc_right[wi][wj][2];
                if (c_base_col + 8 + store_col1 < N)
                    C[(size_t)row_hi * N + c_base_col + 8 + store_col1] = acc_right[wi][wj][3];
            }
        }
    }
}

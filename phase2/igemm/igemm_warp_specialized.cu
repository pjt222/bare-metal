/*
 * igemm_warp_specialized.cu — Producer-consumer warp-specialized IGEMM (Issue #13)
 *
 * Splits 8 warps into 4 IMMA (compute) + 4 quantize (load+convert).
 * Overlaps FP16->INT8 quantization of tile N+1 with IMMA compute on tile N.
 *
 * IMMA warps (0-3): 2x2 grid on 128x128, each covering 64x64.
 *   Each warp's 64x64 region = 4x4 WMMA tiles = 16 tiles.
 *   Split into 2 halves of 2x4 tiles to keep running accumulators at 128 regs.
 *
 * Quant warps (4-7): Issue cp.async for next FP16 tile, wait, two-phase quantize.
 *   Named barrier (bar 2/3, 128 threads) for quant-internal WAR protection.
 *   Named barrier (bar 1, 256 threads) for tile-boundary sync.
 *
 * Smem: FP16 double-buffer (32 KB) + epilogue (8 KB) = 40 KB (same as in-place)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_warp_specialized.sm_86.cubin igemm_warp_specialized.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 32

#define NUM_WARPS   8
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 256 threads

// Prologue uses all 256 threads for cp.async + quantize
#define CP_ASYNC_SIZE 4
#define CP_ELEMS_A  (BM * BK * 2 / BLOCK_SIZE / CP_ASYNC_SIZE)   // 8
#define CP_ELEMS_B  (BK * BN * 2 / BLOCK_SIZE / CP_ASYNC_SIZE)   // 8
#define QUANT_ELEMS_A  (BM * BK / BLOCK_SIZE)   // 16
#define QUANT_ELEMS_B  (BK * BN / BLOCK_SIZE)   // 16

// Main loop: only 128 quant threads do cp.async + quantize
#define QUANT_THREADS  128
#define QCP_ELEMS_A  (BM * BK * 2 / QUANT_THREADS / CP_ASYNC_SIZE)  // 16
#define QCP_ELEMS_B  (BK * BN * 2 / QUANT_THREADS / CP_ASYNC_SIZE)  // 16
#define QQELEMS_A    (BM * BK / QUANT_THREADS)   // 32
#define QQELEMS_B    (BK * BN / QUANT_THREADS)   // 32

// Named barriers (IDs 1-15; 0 is reserved for __syncthreads__)
#define BAR_TILE_BOUNDARY  1   // 256 threads: tile swap sync
#define BAR_CPASYNC_DONE   2   // 128 threads: cp.async data visible
#define BAR_QUANT_PHASE    3   // 128 threads: FP16 reads done before INT8 writes

// ===================================================================
// QUANTIZE_TILE_INPLACE: Full max_abs + two-phase (for prologue, all 256 threads)
// Same as igemm_online_quant_inplace.cu — used only for tile 0.
// ===================================================================
#define QUANTIZE_TILE_INPLACE(BUF)                                             \
do {                                                                           \
    float _max_a = 0.0f;                                                       \
    for (int _i = 0; _i < QUANT_ELEMS_A; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_a_fp16[BUF][_idx]);                       \
        _max_a = fmaxf(_max_a, fabsf(_v));                                     \
    }                                                                          \
    for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1)                        \
        _max_a = fmaxf(_max_a, __shfl_xor_sync(0xFFFFFFFF, _max_a, _off));    \
                                                                               \
    float _max_b = 0.0f;                                                       \
    for (int _i = 0; _i < QUANT_ELEMS_B; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_b_fp16[BUF][_idx]);                       \
        _max_b = fmaxf(_max_b, fabsf(_v));                                     \
    }                                                                          \
    for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1)                        \
        _max_b = fmaxf(_max_b, __shfl_xor_sync(0xFFFFFFFF, _max_b, _off));    \
                                                                               \
    if (lane_id == 0) {                                                        \
        reduce_max[warp_id * 2]     = _max_a;                                  \
        reduce_max[warp_id * 2 + 1] = _max_b;                                 \
    }                                                                          \
    __syncthreads();                                                           \
    float _bma = reduce_max[0];                                                \
    float _bmb = reduce_max[1];                                                \
    for (int _w = 1; _w < NUM_WARPS; _w++) {                                   \
        _bma = fmaxf(_bma, reduce_max[_w * 2]);                               \
        _bmb = fmaxf(_bmb, reduce_max[_w * 2 + 1]);                           \
    }                                                                          \
                                                                               \
    float _inv_sa = (_bma > 0.0f) ? (127.0f / _bma) : 1.0f;                   \
    float _inv_sb = (_bmb > 0.0f) ? (127.0f / _bmb) : 1.0f;                   \
                                                                               \
    signed char _qa[QUANT_ELEMS_A];                                            \
    for (int _i = 0; _i < QUANT_ELEMS_A; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_a_fp16[BUF][_idx]);                       \
        int _q = __float2int_rn(_v * _inv_sa);                                 \
        _q = max(-128, min(127, _q));                                          \
        _qa[_i] = (signed char)_q;                                             \
    }                                                                          \
    signed char _qb[QUANT_ELEMS_B];                                            \
    for (int _i = 0; _i < QUANT_ELEMS_B; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_b_fp16[BUF][_idx]);                       \
        int _q = __float2int_rn(_v * _inv_sb);                                 \
        _q = max(-128, min(127, _q));                                          \
        _qb[_i] = (signed char)_q;                                             \
    }                                                                          \
                                                                               \
    __syncthreads();                                                           \
                                                                               \
    signed char *_dst_a = (signed char *)smem_a_fp16[BUF];                     \
    signed char *_dst_b = (signed char *)smem_b_fp16[BUF];                     \
    for (int _i = 0; _i < QUANT_ELEMS_A; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        _dst_a[_idx] = _qa[_i];                                                \
    }                                                                          \
    for (int _i = 0; _i < QUANT_ELEMS_B; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        _dst_b[_idx] = _qb[_i];                                               \
    }                                                                          \
    __syncthreads();                                                           \
    saved_inv_sa = _inv_sa;                                                    \
    saved_inv_sb = _inv_sb;                                                    \
    tile_scale = (_bma / 127.0f) * (_bmb / 127.0f);                           \
} while(0)


extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void igemm_warp_specialized(
    const __half * __restrict__ matrix_a,   // M*K row-major (FP16)
    const __half * __restrict__ matrix_b,   // K*N row-major (FP16)
    float        * __restrict__ matrix_c,   // M*N row-major (FP32)
    int M, int N, int K
) {
    // ---- Shared memory (40 KB total) ----
    __shared__ __half smem_a_fp16[2][BM * BK];    // 2 * 128*32 = 16 KB
    __shared__ __half smem_b_fp16[2][BK * BN];    // 2 * 32*128 = 16 KB
    __shared__ float epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 KB
    float *reduce_max = epilogue_tile[0];

    // ---- Thread indices ----
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    bool is_imma_warp = (warp_id < 4);

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int num_tiles = (K + BK - 1) / BK;

    float tile_scale   = 0.0f;
    float saved_inv_sa = 1.0f;
    float saved_inv_sb = 1.0f;

    // ====================================================================
    // PROLOGUE: All 256 threads cooperate to load + quantize tile 0
    // ====================================================================
    #pragma unroll
    for (int i = 0; i < CP_ELEMS_A; i++) {
        int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
        int elem_idx = byte_idx / 2;
        int row      = elem_idx / BK;
        int col      = elem_idx % BK;
        int g_row    = block_row + row;
        int g_col    = col;

        if (g_row < M && (g_col + 1) < K) {
            __pipeline_memcpy_async(
                (char*)smem_a_fp16[0] + byte_idx,
                (const char*)(matrix_a + g_row * K + g_col),
                CP_ASYNC_SIZE);
        } else {
            __half *dst = smem_a_fp16[0] + elem_idx;
            for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                int gc = g_col + b;
                dst[b] = (g_row < M && gc < K) ? matrix_a[g_row * K + gc] : __float2half(0.0f);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CP_ELEMS_B; i++) {
        int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
        int elem_idx = byte_idx / 2;
        int row      = elem_idx / BN;
        int col      = elem_idx % BN;
        int g_row    = row;
        int g_col    = block_col + col;

        if (g_row < K && (g_col + 1) < N) {
            __pipeline_memcpy_async(
                (char*)smem_b_fp16[0] + byte_idx,
                (const char*)(matrix_b + g_row * N + g_col),
                CP_ASYNC_SIZE);
        } else {
            __half *dst = smem_b_fp16[0] + elem_idx;
            for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                int gc = g_col + b;
                dst[b] = (g_row < K && gc < N) ? matrix_b[g_row * N + gc] : __float2half(0.0f);
            }
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Quantize tile 0 in-place (full max_abs, all 256 threads)
    QUANTIZE_TILE_INPLACE(0);

    // ====================================================================
    // MAIN LOOP: Warp-specialized — IMMA || load+quantize
    // ====================================================================

    if (is_imma_warp) {
        // ---- IMMA warp path (warps 0-3) ----
        // 2x2 grid on 128x128: each warp covers 64x64
        int imma_wy = warp_id / 2;   // 0 or 1
        int imma_wx = warp_id % 2;   // 0 or 1

        // Running accumulators: [2 halves][2 tile_m][4 tile_n][8 elements]
        // Half 0 = top 32 rows of warp's 64-row region
        // Half 1 = bottom 32 rows
        float running[2][2][4][8];
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    #pragma unroll
                    for (int e = 0; e < 8; e++)
                        running[h][i][j][e] = 0.0f;

        // ---- Main K-tile loop ----
        for (int tile = 0; tile < num_tiles - 1; tile++) {
            int cur_buf = tile & 1;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
                acc[2][4];

            // ---- Half 0: top 32 rows [imma_wy*64 .. imma_wy*64+31] ----
            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    wmma::fill_fragment(acc[wi][wj], 0);

            #pragma unroll
            for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> a_frag[2];
                #pragma unroll
                for (int wi = 0; wi < 2; wi++) {
                    int a_row = imma_wy * 64 + wi * WMMA_M;  // half 0: rows 0-31
                    wmma::load_matrix_sync(a_frag[wi],
                        (const signed char *)smem_a_fp16[cur_buf] + a_row * BK + k_local, BK);
                }

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> b_frag[4];
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int b_col = imma_wx * 64 + wj * WMMA_N;
                    wmma::load_matrix_sync(b_frag[wj],
                        (const signed char *)smem_b_fp16[cur_buf] + k_local * BN + b_col, BN);
                }

                #pragma unroll
                for (int wi = 0; wi < 2; wi++)
                    #pragma unroll
                    for (int wj = 0; wj < 4; wj++)
                        wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
            }

            // Dequant half 0
            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    #pragma unroll
                    for (int e = 0; e < 8; e++)
                        running[0][wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;

            // ---- Half 1: bottom 32 rows [imma_wy*64+32 .. imma_wy*64+63] ----
            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    wmma::fill_fragment(acc[wi][wj], 0);

            #pragma unroll
            for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> a_frag[2];
                #pragma unroll
                for (int wi = 0; wi < 2; wi++) {
                    int a_row = imma_wy * 64 + 32 + wi * WMMA_M;  // half 1: rows 32-63
                    wmma::load_matrix_sync(a_frag[wi],
                        (const signed char *)smem_a_fp16[cur_buf] + a_row * BK + k_local, BK);
                }

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> b_frag[4];
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int b_col = imma_wx * 64 + wj * WMMA_N;
                    wmma::load_matrix_sync(b_frag[wj],
                        (const signed char *)smem_b_fp16[cur_buf] + k_local * BN + b_col, BN);
                }

                #pragma unroll
                for (int wi = 0; wi < 2; wi++)
                    #pragma unroll
                    for (int wj = 0; wj < 4; wj++)
                        wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
            }

            // Dequant half 1
            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    #pragma unroll
                    for (int e = 0; e < 8; e++)
                        running[1][wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;

            // ---- Tile boundary barrier: wait for quant warps ----
            asm volatile("bar.sync %0, %1;" :: "r"(BAR_TILE_BOUNDARY), "r"(BLOCK_SIZE));
        }

        // ====================================================================
        // LAST K-TILE: IMMA only (no more loading/quantizing)
        // ====================================================================
        {
            int last_buf = (num_tiles - 1) & 1;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
                acc[2][4];

            // Half 0
            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    wmma::fill_fragment(acc[wi][wj], 0);

            #pragma unroll
            for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> a_frag[2];
                #pragma unroll
                for (int wi = 0; wi < 2; wi++) {
                    int a_row = imma_wy * 64 + wi * WMMA_M;
                    wmma::load_matrix_sync(a_frag[wi],
                        (const signed char *)smem_a_fp16[last_buf] + a_row * BK + k_local, BK);
                }

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> b_frag[4];
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int b_col = imma_wx * 64 + wj * WMMA_N;
                    wmma::load_matrix_sync(b_frag[wj],
                        (const signed char *)smem_b_fp16[last_buf] + k_local * BN + b_col, BN);
                }

                #pragma unroll
                for (int wi = 0; wi < 2; wi++)
                    #pragma unroll
                    for (int wj = 0; wj < 4; wj++)
                        wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
            }

            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    #pragma unroll
                    for (int e = 0; e < 8; e++)
                        running[0][wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;

            // Half 1
            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    wmma::fill_fragment(acc[wi][wj], 0);

            #pragma unroll
            for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> a_frag[2];
                #pragma unroll
                for (int wi = 0; wi < 2; wi++) {
                    int a_row = imma_wy * 64 + 32 + wi * WMMA_M;
                    wmma::load_matrix_sync(a_frag[wi],
                        (const signed char *)smem_a_fp16[last_buf] + a_row * BK + k_local, BK);
                }

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> b_frag[4];
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int b_col = imma_wx * 64 + wj * WMMA_N;
                    wmma::load_matrix_sync(b_frag[wj],
                        (const signed char *)smem_b_fp16[last_buf] + k_local * BN + b_col, BN);
                }

                #pragma unroll
                for (int wi = 0; wi < 2; wi++)
                    #pragma unroll
                    for (int wj = 0; wj < 4; wj++)
                        wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
            }

            #pragma unroll
            for (int wi = 0; wi < 2; wi++)
                #pragma unroll
                for (int wj = 0; wj < 4; wj++)
                    #pragma unroll
                    for (int e = 0; e < 8; e++)
                        running[1][wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;
        }

        // ====================================================================
        // EPILOGUE: Write both halves of running accumulators to global
        // ====================================================================
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            #pragma unroll
            for (int wi = 0; wi < 2; wi++) {
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int c_row = block_row + imma_wy * 64 + half * 32 + wi * WMMA_M;
                    int c_col = block_col + imma_wx * 64 + wj * WMMA_N;

                    if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;
                    #pragma unroll
                    for (int e = 0; e < 8; e++)
                        out_frag.x[e] = running[half][wi][wj][e];

                    wmma::store_matrix_sync(
                        epilogue_tile[warp_id],
                        out_frag, WMMA_N, wmma::mem_row_major);
                    __syncwarp();

                    for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += WARP_SIZE) {
                        int local_row = elem / WMMA_N;
                        int local_col = elem % WMMA_N;
                        matrix_c[(c_row + local_row) * N + (c_col + local_col)] =
                            epilogue_tile[warp_id][elem];
                    }
                    __syncwarp();
                }
            }
        }

    } else {
        // ---- Quant warp path (warps 4-7) ----
        int qtid = tid - QUANT_THREADS;  // 0-127 within quant group

        for (int tile = 0; tile < num_tiles - 1; tile++) {
            int next_k_base = (tile + 1) * BK;
            int next_buf    = 1 - (tile & 1);

            // ---- cp.async: load next FP16 tile (128 quant threads) ----
            #pragma unroll
            for (int i = 0; i < QCP_ELEMS_A; i++) {
                int byte_idx = (qtid + i * QUANT_THREADS) * CP_ASYNC_SIZE;
                int elem_idx = byte_idx / 2;
                int row      = elem_idx / BK;
                int col      = elem_idx % BK;
                int g_row    = block_row + row;
                int g_col    = next_k_base + col;

                if (g_row < M && (g_col + 1) < K) {
                    __pipeline_memcpy_async(
                        (char*)smem_a_fp16[next_buf] + byte_idx,
                        (const char*)(matrix_a + g_row * K + g_col),
                        CP_ASYNC_SIZE);
                } else {
                    __half *dst = smem_a_fp16[next_buf] + elem_idx;
                    for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                        int gc = g_col + b;
                        dst[b] = (g_row < M && gc < K) ? matrix_a[g_row * K + gc] : __float2half(0.0f);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < QCP_ELEMS_B; i++) {
                int byte_idx = (qtid + i * QUANT_THREADS) * CP_ASYNC_SIZE;
                int elem_idx = byte_idx / 2;
                int row      = elem_idx / BN;
                int col      = elem_idx % BN;
                int g_row    = next_k_base + row;
                int g_col    = block_col + col;

                if (g_row < K && (g_col + 1) < N) {
                    __pipeline_memcpy_async(
                        (char*)smem_b_fp16[next_buf] + byte_idx,
                        (const char*)(matrix_b + g_row * N + g_col),
                        CP_ASYNC_SIZE);
                } else {
                    __half *dst = smem_b_fp16[next_buf] + elem_idx;
                    for (int b = 0; b < CP_ASYNC_SIZE / 2; b++) {
                        int gc = g_col + b;
                        dst[b] = (g_row < K && gc < N) ? matrix_b[g_row * N + gc] : __float2half(0.0f);
                    }
                }
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);

            // ---- Barrier: all quant threads' cp.async data visible ----
            asm volatile("bar.sync %0, %1;" :: "r"(BAR_CPASYNC_DONE), "r"(QUANT_THREADS));

            // ---- Two-phase quantize (quant warps only) ----
            // Phase 1: Read FP16 from next_buf, convert to INT8 in registers
            signed char qa[QQELEMS_A];
            #pragma unroll
            for (int i = 0; i < QQELEMS_A; i++) {
                int idx = qtid + i * QUANT_THREADS;
                float v = __half2float(smem_a_fp16[next_buf][idx]);
                int q = __float2int_rn(v * saved_inv_sa);
                q = max(-128, min(127, q));
                qa[i] = (signed char)q;
            }
            signed char qb[QQELEMS_B];
            #pragma unroll
            for (int i = 0; i < QQELEMS_B; i++) {
                int idx = qtid + i * QUANT_THREADS;
                float v = __half2float(smem_b_fp16[next_buf][idx]);
                int q = __float2int_rn(v * saved_inv_sb);
                q = max(-128, min(127, q));
                qb[i] = (signed char)q;
            }

            // Phase barrier: all FP16 reads done before any INT8 writes (WAR protection)
            asm volatile("bar.sync %0, %1;" :: "r"(BAR_QUANT_PHASE), "r"(QUANT_THREADS));

            // Phase 2: Write INT8 in-place
            signed char *dst_a = (signed char *)smem_a_fp16[next_buf];
            signed char *dst_b = (signed char *)smem_b_fp16[next_buf];
            #pragma unroll
            for (int i = 0; i < QQELEMS_A; i++) {
                int idx = qtid + i * QUANT_THREADS;
                dst_a[idx] = qa[i];
            }
            #pragma unroll
            for (int i = 0; i < QQELEMS_B; i++) {
                int idx = qtid + i * QUANT_THREADS;
                dst_b[idx] = qb[i];
            }

            // ---- Tile boundary barrier: signal next_buf ready to IMMA warps ----
            asm volatile("bar.sync %0, %1;" :: "r"(BAR_TILE_BOUNDARY), "r"(BLOCK_SIZE));
        }
        // After main loop: quant warps have nothing to do.
        // IMMA warps handle last tile + epilogue independently.
    }
}

/*
 * igemm_online_quant_bankfree.cu — Bank-conflict-free FP16->INT8 quantized GEMM
 *
 * Same algorithm as igemm_online_quant_inplace.cu but with vectorized smem
 * access patterns in the quantize macros to eliminate bank conflicts:
 *
 *   FP16 reads:  __half2 loads (4-byte, stride-256) → 0-way conflicts (was 2-way)
 *   INT8 writes: packed uint32 stores (4-byte, stride-256) → 0-way conflicts (was 4-way)
 *
 * The grouped-4 read pattern (2 consecutive __half2 per thread per iteration)
 * has 2-way conflicts per load, but the uint32 write is conflict-free.
 * Net: ~6× reduction in bank-conflict overhead (192 → 32 extra passes per tile).
 *
 * Smem: unchanged (32 KB FP16 double-buffer + 8 KB epilogue = 40 KB)
 * Expected: +10-15% over in-place baseline (which ran 16,646 GFLOPS at 4096³)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_online_quant_bankfree.sm_86.cubin igemm_online_quant_bankfree.cu
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

#define WARPS_Y 4
#define WARPS_X 2

#define WARP_TILES_M 2   // 32 / 16
#define WARP_TILES_N 4   // 64 / 16

#define CP_ASYNC_SIZE 4
#define CP_ELEMS_A  (BM * BK * 2 / BLOCK_SIZE / CP_ASYNC_SIZE)   // 8
#define CP_ELEMS_B  (BK * BN * 2 / BLOCK_SIZE / CP_ASYNC_SIZE)   // 8

#define QUANT_ELEMS_A  (BM * BK / BLOCK_SIZE)   // 16
#define QUANT_ELEMS_B  (BK * BN / BLOCK_SIZE)   // 16

// ===================================================================
// QUANTIZE_TILE_INPLACE_BF: Bank-conflict-free full quantize (prologue).
//
// Max_abs pass: stride-256 __half2 reads (0 conflicts).
// Conversion pass: grouped-4 __half2 pair reads (2-way) + uint32 writes (0 conflicts).
// Only used for first K-tile; subsequent tiles use FAST variant.
// ===================================================================
#define QUANTIZE_TILE_INPLACE_BF(BUF)                                          \
do {                                                                           \
    /* --- Max_abs pass: stride-256 __half2 reads (0-conflict) --- */          \
    float _max_a = 0.0f;                                                       \
    {                                                                          \
        __half2 *_src_a = (__half2 *)smem_a_fp16[BUF];                         \
        for (int _i = 0; _i < QUANT_ELEMS_A / 2; _i++) {                      \
            __half2 _v2 = _src_a[tid + _i * BLOCK_SIZE];                       \
            _max_a = fmaxf(_max_a, fmaxf(fabsf(__low2float(_v2)),              \
                                         fabsf(__high2float(_v2))));           \
        }                                                                      \
    }                                                                          \
    for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1)                        \
        _max_a = fmaxf(_max_a, __shfl_xor_sync(0xFFFFFFFF, _max_a, _off));    \
                                                                               \
    float _max_b = 0.0f;                                                       \
    {                                                                          \
        __half2 *_src_b = (__half2 *)smem_b_fp16[BUF];                         \
        for (int _i = 0; _i < QUANT_ELEMS_B / 2; _i++) {                      \
            __half2 _v2 = _src_b[tid + _i * BLOCK_SIZE];                       \
            _max_b = fmaxf(_max_b, fmaxf(fabsf(__low2float(_v2)),              \
                                         fabsf(__high2float(_v2))));           \
        }                                                                      \
    }                                                                          \
    for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1)                        \
        _max_b = fmaxf(_max_b, __shfl_xor_sync(0xFFFFFFFF, _max_b, _off));    \
                                                                               \
    if (lane_id == 0) {                                                        \
        reduce_max[warp_id * 2]     = _max_a;                                  \
        reduce_max[warp_id * 2 + 1] = _max_b;                                 \
    }                                                                          \
    __syncthreads();  /* sync1: cross-warp reduction */                        \
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
    /* --- Conversion: grouped-4 __half2 reads + register store --- */         \
    signed char _qa[QUANT_ELEMS_A];                                            \
    {                                                                          \
        __half2 *_src_a = (__half2 *)smem_a_fp16[BUF];                         \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_A / 4; _i++) {                      \
            int _base = (tid + _i * BLOCK_SIZE) * 2;                           \
            __half2 _v0 = _src_a[_base];                                       \
            __half2 _v1 = _src_a[_base + 1];                                   \
            int _q0 = __float2int_rn(__low2float(_v0) * _inv_sa);              \
            int _q1 = __float2int_rn(__high2float(_v0) * _inv_sa);             \
            int _q2 = __float2int_rn(__low2float(_v1) * _inv_sa);              \
            int _q3 = __float2int_rn(__high2float(_v1) * _inv_sa);             \
            _qa[_i*4+0] = (signed char)max(-128, min(127, _q0));               \
            _qa[_i*4+1] = (signed char)max(-128, min(127, _q1));               \
            _qa[_i*4+2] = (signed char)max(-128, min(127, _q2));               \
            _qa[_i*4+3] = (signed char)max(-128, min(127, _q3));               \
        }                                                                      \
    }                                                                          \
    signed char _qb[QUANT_ELEMS_B];                                            \
    {                                                                          \
        __half2 *_src_b = (__half2 *)smem_b_fp16[BUF];                         \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_B / 4; _i++) {                      \
            int _base = (tid + _i * BLOCK_SIZE) * 2;                           \
            __half2 _v0 = _src_b[_base];                                       \
            __half2 _v1 = _src_b[_base + 1];                                   \
            int _q0 = __float2int_rn(__low2float(_v0) * _inv_sb);              \
            int _q1 = __float2int_rn(__high2float(_v0) * _inv_sb);             \
            int _q2 = __float2int_rn(__low2float(_v1) * _inv_sb);              \
            int _q3 = __float2int_rn(__high2float(_v1) * _inv_sb);             \
            _qb[_i*4+0] = (signed char)max(-128, min(127, _q0));               \
            _qb[_i*4+1] = (signed char)max(-128, min(127, _q1));               \
            _qb[_i*4+2] = (signed char)max(-128, min(127, _q2));               \
            _qb[_i*4+3] = (signed char)max(-128, min(127, _q3));               \
        }                                                                      \
    }                                                                          \
                                                                               \
    __syncthreads();  /* sync2: all reads done before any writes */            \
                                                                               \
    /* --- Write INT8 as packed uint32 (0 conflicts) --- */                    \
    {                                                                          \
        unsigned int *_dst_a32 = (unsigned int *)                              \
            ((signed char *)smem_a_fp16[BUF]);                                 \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_A / 4; _i++) {                      \
            int _idx = tid + _i * BLOCK_SIZE;                                  \
            unsigned int _p = ((unsigned int)(unsigned char)_qa[_i*4+0])        \
                | ((unsigned int)(unsigned char)_qa[_i*4+1] << 8)              \
                | ((unsigned int)(unsigned char)_qa[_i*4+2] << 16)             \
                | ((unsigned int)(unsigned char)_qa[_i*4+3] << 24);            \
            _dst_a32[_idx] = _p;                                               \
        }                                                                      \
        unsigned int *_dst_b32 = (unsigned int *)                              \
            ((signed char *)smem_b_fp16[BUF]);                                 \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_B / 4; _i++) {                      \
            int _idx = tid + _i * BLOCK_SIZE;                                  \
            unsigned int _p = ((unsigned int)(unsigned char)_qb[_i*4+0])        \
                | ((unsigned int)(unsigned char)_qb[_i*4+1] << 8)              \
                | ((unsigned int)(unsigned char)_qb[_i*4+2] << 16)             \
                | ((unsigned int)(unsigned char)_qb[_i*4+3] << 24);            \
            _dst_b32[_idx] = _p;                                               \
        }                                                                      \
    }                                                                          \
    __syncthreads();  /* sync3: all writes visible before IMMA reads */        \
    saved_inv_sa = _inv_sa;                                                    \
    saved_inv_sb = _inv_sb;                                                    \
    tile_scale = (_bma / 127.0f) * (_bmb / 127.0f);                           \
} while(0)

// ===================================================================
// QUANTIZE_TILE_FAST_BF: Bank-conflict-free fast quantize (main loop).
//
// Grouped-4 pattern: read 2 consecutive __half2 (2-way conflict per load),
// convert 4 FP16→INT8, write as packed uint32 (0 conflicts).
// ~6x fewer bank-conflict stalls than the original byte-wise pattern.
// ===================================================================
#define QUANTIZE_TILE_FAST_BF(BUF)                                             \
do {                                                                           \
    /* Phase 1: Read FP16 as __half2 pairs, convert to INT8 in regs */        \
    signed char _qa[QUANT_ELEMS_A];                                            \
    {                                                                          \
        __half2 *_src_a = (__half2 *)smem_a_fp16[BUF];                         \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_A / 4; _i++) {                      \
            int _base = (tid + _i * BLOCK_SIZE) * 2;                           \
            __half2 _v0 = _src_a[_base];                                       \
            __half2 _v1 = _src_a[_base + 1];                                   \
            int _q0 = __float2int_rn(__low2float(_v0) * saved_inv_sa);         \
            int _q1 = __float2int_rn(__high2float(_v0) * saved_inv_sa);        \
            int _q2 = __float2int_rn(__low2float(_v1) * saved_inv_sa);         \
            int _q3 = __float2int_rn(__high2float(_v1) * saved_inv_sa);        \
            _qa[_i*4+0] = (signed char)max(-128, min(127, _q0));               \
            _qa[_i*4+1] = (signed char)max(-128, min(127, _q1));               \
            _qa[_i*4+2] = (signed char)max(-128, min(127, _q2));               \
            _qa[_i*4+3] = (signed char)max(-128, min(127, _q3));               \
        }                                                                      \
    }                                                                          \
    signed char _qb[QUANT_ELEMS_B];                                            \
    {                                                                          \
        __half2 *_src_b = (__half2 *)smem_b_fp16[BUF];                         \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_B / 4; _i++) {                      \
            int _base = (tid + _i * BLOCK_SIZE) * 2;                           \
            __half2 _v0 = _src_b[_base];                                       \
            __half2 _v1 = _src_b[_base + 1];                                   \
            int _q0 = __float2int_rn(__low2float(_v0) * saved_inv_sb);         \
            int _q1 = __float2int_rn(__high2float(_v0) * saved_inv_sb);        \
            int _q2 = __float2int_rn(__low2float(_v1) * saved_inv_sb);         \
            int _q3 = __float2int_rn(__high2float(_v1) * saved_inv_sb);        \
            _qb[_i*4+0] = (signed char)max(-128, min(127, _q0));               \
            _qb[_i*4+1] = (signed char)max(-128, min(127, _q1));               \
            _qb[_i*4+2] = (signed char)max(-128, min(127, _q2));               \
            _qb[_i*4+3] = (signed char)max(-128, min(127, _q3));               \
        }                                                                      \
    }                                                                          \
                                                                               \
    __syncthreads();  /* sync1: all reads done before any writes */            \
                                                                               \
    /* Phase 2: Write INT8 as packed uint32 (0 conflicts) */                   \
    {                                                                          \
        unsigned int *_dst_a32 = (unsigned int *)                              \
            ((signed char *)smem_a_fp16[BUF]);                                 \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_A / 4; _i++) {                      \
            int _idx = tid + _i * BLOCK_SIZE;                                  \
            unsigned int _p = ((unsigned int)(unsigned char)_qa[_i*4+0])        \
                | ((unsigned int)(unsigned char)_qa[_i*4+1] << 8)              \
                | ((unsigned int)(unsigned char)_qa[_i*4+2] << 16)             \
                | ((unsigned int)(unsigned char)_qa[_i*4+3] << 24);            \
            _dst_a32[_idx] = _p;                                               \
        }                                                                      \
        unsigned int *_dst_b32 = (unsigned int *)                              \
            ((signed char *)smem_b_fp16[BUF]);                                 \
        _Pragma("unroll")                                                      \
        for (int _i = 0; _i < QUANT_ELEMS_B / 4; _i++) {                      \
            int _idx = tid + _i * BLOCK_SIZE;                                  \
            unsigned int _p = ((unsigned int)(unsigned char)_qb[_i*4+0])        \
                | ((unsigned int)(unsigned char)_qb[_i*4+1] << 8)              \
                | ((unsigned int)(unsigned char)_qb[_i*4+2] << 16)             \
                | ((unsigned int)(unsigned char)_qb[_i*4+3] << 24);            \
            _dst_b32[_idx] = _p;                                               \
        }                                                                      \
    }                                                                          \
    __syncthreads();  /* sync2: all writes visible before IMMA reads */        \
} while(0)


extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void igemm_online_quant_bankfree(
    const __half * __restrict__ matrix_a,   // M*K row-major (FP16)
    const __half * __restrict__ matrix_b,   // K*N row-major (FP16)
    float        * __restrict__ matrix_c,   // M*N row-major (FP32)
    int M, int N, int K
) {
    // FP16 double-buffer — INT8 written in-place to lower half
    __shared__ __half smem_a_fp16[2][BM * BK];    // 2 * 128*32 = 16 KB
    __shared__ __half smem_b_fp16[2][BK * BN];    // 2 * 32*128 = 16 KB

    // Epilogue + cross-warp reduction scratch
    __shared__ float epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 KB
    float *reduce_max = epilogue_tile[0];

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;
    int wx      = warp_id % WARPS_X;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int num_tiles = (K + BK - 1) / BK;

    // INT32 IMMA accumulators (reset per K-tile)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
        acc[WARP_TILES_M][WARP_TILES_N];

    // FP32 running accumulators (persist across K-tiles)
    float running[WARP_TILES_M][WARP_TILES_N][8];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++)
                running[i][j][e] = 0.0f;

    float tile_scale = 0.0f;
    float saved_inv_sa = 1.0f;
    float saved_inv_sb = 1.0f;

    // ====================================================================
    // Prologue: cp.async FP16 tile 0 -> buf 0, wait, quantize in-place
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

    // Quantize tile 0 in-place (bank-conflict-free, full max_abs)
    QUANTIZE_TILE_INPLACE_BF(0);

    // ====================================================================
    // Main loop: load FP16(N+1), IMMA on in-place INT8(N), dequant, quantize(N+1)
    // ====================================================================
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        // --- Zero INT32 accumulators ---
        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; i++)
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++)
                wmma::fill_fragment(acc[i][j], 0);

        // --- Phase 1: cp.async for NEXT FP16 tile ---
        #pragma unroll
        for (int i = 0; i < CP_ELEMS_A; i++) {
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
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
        for (int i = 0; i < CP_ELEMS_B; i++) {
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
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

        // --- Phase 2: IMMA on current tile (in-place INT8 in FP16 buffer) ---
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    (const signed char *)smem_a_fp16[cur_buf] + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    (const signed char *)smem_b_fp16[cur_buf] + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        // --- Phase 3: Dequantize INT32 -> FP32 running accumulator ---
        #pragma unroll
        for (int wi = 0; wi < WARP_TILES_M; wi++)
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    running[wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;

        // --- Phase 4: Wait for cp.async, quantize next tile (bank-free) ---
        __pipeline_wait_prior(0);
        __syncthreads();

        QUANTIZE_TILE_FAST_BF(next_buf);
    }

    // ====================================================================
    // Last K-tile: IMMA + final dequant
    // ====================================================================
    {
        int last_buf = (num_tiles - 1) & 1;

        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; i++)
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++)
                wmma::fill_fragment(acc[i][j], 0);

        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    (const signed char *)smem_a_fp16[last_buf] + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    (const signed char *)smem_b_fp16[last_buf] + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        // Final dequant
        #pragma unroll
        for (int wi = 0; wi < WARP_TILES_M; wi++)
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    running[wi][wj][e] += (float)acc[wi][wj].x[e] * tile_scale;
    }

    // ====================================================================
    // Store FP32 running accumulators via smem epilogue
    // ====================================================================
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 64 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;
            #pragma unroll
            for (int e = 0; e < 8; e++)
                out_frag.x[e] = running[wi][wj][e];

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

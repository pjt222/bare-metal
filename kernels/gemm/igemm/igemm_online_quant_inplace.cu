/*
 * igemm_online_quant_inplace.cu — In-place FP16->INT8 quantized GEMM (Issue #15 fix)
 *
 * Same as igemm_online_quant.cu but eliminates 8 KB of shared memory by writing
 * INT8 data directly into the lower half of the FP16 double-buffer.
 *
 * Root cause of original failure: cross-warp WAR (Write-After-Read) hazard.
 * Thread T writes INT8 at byte `idx`, corrupting FP16 element `idx/2`.
 * Within a warp, SIMT lockstep ensures read-before-write.
 * Across warps, no such guarantee — warp scheduling is non-deterministic.
 *
 * Fix: Two-phase quantize — ALL threads read FP16 to registers, __syncthreads(),
 * then ALL threads write INT8. Extra cost: 1 sync per K-tile (~3 us total at 4096K)
 * and 8 extra registers (238 total, under 255 limit).
 *
 * Smem: FP16 double-buffer (32 KB) + epilogue (8 KB) = 40 KB (was 48 KB)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_online_quant_inplace.sm_86.cubin igemm_online_quant_inplace.cu
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
// QUANTIZE_TILE_INPLACE: Two-phase quantize with read-write barrier.
//
// Phase 1: Read FP16, compute max_abs (warp reduce + cross-warp reduce).
// Phase 2: Re-read FP16, convert to INT8, store in registers.
//          (Safe: no INT8 writes have happened yet, FP16 data is intact.)
// Barrier: __syncthreads() — all reads complete before any writes.
// Phase 3: Write INT8 in-place to the FP16 buffer.
// Barrier: __syncthreads() — all writes visible before IMMA reads.
//
// Syncs: 3 (1 for cross-warp reduce, 1 read-write barrier, 1 write-done)
// vs. 2 in the separate-buffer version (no read-write barrier needed).
// Only used for first K-tile; subsequent tiles use FAST variant.
// ===================================================================
#define QUANTIZE_TILE_INPLACE(BUF)                                             \
do {                                                                           \
    /* --- Phase 1: max_abs reduction (reads FP16, doesn't store values) --- */\
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
    /* --- Phase 2: Re-read FP16, convert to INT8, store in regs --- */        \
    /* FP16 data is still intact — no INT8 writes have happened yet. */        \
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
    __syncthreads();  /* sync2: all reads done before any writes */            \
                                                                               \
    /* --- Phase 3: Write INT8 in-place to lower half of FP16 buffer --- */    \
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
    __syncthreads();  /* sync3: all writes visible before IMMA reads */        \
    saved_inv_sa = _inv_sa;                                                    \
    saved_inv_sb = _inv_sb;                                                    \
    tile_scale = (_bma / 127.0f) * (_bmb / 127.0f);                           \
} while(0)

// ===================================================================
// QUANTIZE_TILE_FAST_INPLACE: Two-phase, skip max_abs, reuse saved scales.
//
// Phase 1: Read FP16, convert to INT8, store in registers.
// Barrier: __syncthreads() — all reads complete.
// Phase 2: Write INT8 in-place.
// Barrier: __syncthreads() — all writes visible.
//
// Syncs: 2 (vs. 1 in the separate-buffer version).
// Extra sync cost at 4096K: 127 tiles * ~40 cycles = ~3 us on 10.8 ms kernel.
// ===================================================================
#define QUANTIZE_TILE_FAST_INPLACE(BUF)                                        \
do {                                                                           \
    /* Phase 1: Read + convert to registers */                                 \
    signed char _qa[QUANT_ELEMS_A];                                            \
    for (int _i = 0; _i < QUANT_ELEMS_A; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_a_fp16[BUF][_idx]);                       \
        int _q = __float2int_rn(_v * saved_inv_sa);                            \
        _q = max(-128, min(127, _q));                                          \
        _qa[_i] = (signed char)_q;                                             \
    }                                                                          \
    signed char _qb[QUANT_ELEMS_B];                                            \
    for (int _i = 0; _i < QUANT_ELEMS_B; _i++) {                              \
        int _idx = tid + _i * BLOCK_SIZE;                                      \
        float _v = __half2float(smem_b_fp16[BUF][_idx]);                       \
        int _q = __float2int_rn(_v * saved_inv_sb);                            \
        _q = max(-128, min(127, _q));                                          \
        _qb[_i] = (signed char)_q;                                             \
    }                                                                          \
                                                                               \
    __syncthreads();  /* sync1: all reads done before any writes */            \
                                                                               \
    /* Phase 2: Write INT8 in-place */                                         \
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
    __syncthreads();  /* sync2: all writes visible before IMMA reads */        \
} while(0)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void igemm_online_quant_inplace(
    const __half * __restrict__ matrix_a,   // M*K row-major (FP16)
    const __half * __restrict__ matrix_b,   // K*N row-major (FP16)
    float        * __restrict__ matrix_c,   // M*N row-major (FP32)
    int M, int N, int K
) {
    // FP16 double-buffer for cp.async — INT8 data written in-place to lower half
    __shared__ __half smem_a_fp16[2][BM * BK];    // 2 * 128*32 = 16 KB
    __shared__ __half smem_b_fp16[2][BK * BN];    // 2 * 32*128 = 16 KB

    // No separate INT8 buffers — saved 8 KB (48 KB -> 40 KB)

    // Epilogue: store FP32 accumulator for coalesced writes
    __shared__ float epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 KB

    // Cross-warp reduction scratch — alias into epilogue_tile
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

    // Quantize tile 0 in-place (with full max_abs reduction)
    QUANTIZE_TILE_INPLACE(0);

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

        // --- Phase 4: Wait for cp.async, quantize next tile in-place ---
        __pipeline_wait_prior(0);
        __syncthreads();

        QUANTIZE_TILE_FAST_INPLACE(next_buf);
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

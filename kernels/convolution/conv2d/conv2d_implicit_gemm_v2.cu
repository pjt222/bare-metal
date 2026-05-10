/*
 * conv2d_implicit_gemm_v2.cu — 16-warp 128x128x32 implicit GEMM + cp.async
 *
 * Direct port of hgemm_16warp.cu's tile structure with implicit-GEMM
 * coordinate tables added. Targets the ResBlock outlier identified in
 * Observation BB (1.2% peak, TC util 3.19% on the v1 kernel).
 *
 * Differences vs v1 (conv2d_implicit_gemm.cu):
 *   v1: 4 warps,  64x64x16,  no cp.async,  FP32 input (inline cast)
 *   v2: 16 warps, 128x128x32, cp.async double-buffered, FP16 input
 *
 * Why FP16 input: cp.async cannot perform FP32->FP16 conversion mid-flight,
 * so we require the caller to pre-cast X to FP16. For the ResBlock
 * pipeline, this is fused with groupnorm_silu_fused_fp16 (separate kernel)
 * or done as a small standalone pass.
 *
 * Tile structure (matches hgemm_16warp):
 *   Block: 128 rows of M (im2col output rows), 128 cols of N (Cout),
 *          K-step of 32 (im2col input columns = Cin*kH*kW indices)
 *   Warps: 4x4 grid, each warp covers 32x32 = 2x2 WMMA tiles
 *   Smem:  2 buffers x 128 * (32+8) halfs (A) + 2 x 32 * (128+8) halfs (B)
 *          = 20 KB + 17 KB = 37 KB (under the 50 KB cliff)
 *   2 blocks/SM via __launch_bounds__(512, 2)
 *
 * Coordinate tables:
 *   smem_m_n[128]: n_idx for each M-row in this block
 *   smem_m_oh[128]: out_h for each M-row
 *   smem_m_ow[128]: out_w for each M-row
 *   smem_k_cin[32]: cin per K-col (refilled per K-tile)
 *   smem_k_kh[32]:  kh per K-col
 *   smem_k_kw[32]:  kw per K-col
 *   Total table memory: 3*128*4 + 3*32*4 = 1536+384 = 1920 bytes ≈ 2 KB.
 *   Grand total smem: ~39 KB still under 50 KB.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o conv2d_implicit_gemm_v2.sm_86.cubin conv2d_implicit_gemm_v2.cu
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

#define PAD_A    8
#define PAD_B    8
#define STRIDE_A (BK + PAD_A)
#define STRIDE_B (BN + PAD_B)

#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)   // 512 threads

#define WARPS_Y 4
#define WARPS_X 4

#define WARP_TILES_M 2
#define WARP_TILES_N 2

#define CP_ASYNC_BYTES  16
#define ELEMS_PER_COPY  (CP_ASYNC_BYTES / 2)
// BM*BK = 4096 halfs total for A; per thread = 8 halfs = 1 cp.async call.
#define CP_ELEMS_A      ((BM * BK) / (BLOCK_SIZE * ELEMS_PER_COPY))   // 1
#define CP_ELEMS_B      ((BK * BN) / (BLOCK_SIZE * ELEMS_PER_COPY))   // 1

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void implicit_gemm_conv_v2(
    const __half * __restrict__ X,    // [N, H_in, W_in, Cin]  FP16 NHWC input
    const __half * __restrict__ B,    // [K_dim, Cout]         FP16 weights
    float        * __restrict__ Y,    // [M, Cout]             FP32 output
    int N_batch, int H_in, int W_in, int Cin,
    int kH, int kW, int pad,
    int out_H, int out_W,
    int M, int K_dim, int Cout
) {
    // Shared memory layout
    __shared__ __align__(16) __half smem_a[2][BM * STRIDE_A];   // 20 KB
    __shared__ __align__(16) __half smem_b[2][BK * STRIDE_B];   // 17 KB
    __shared__ int smem_m_n [BM];
    __shared__ int smem_m_oh[BM];
    __shared__ int smem_m_ow[BM];
    __shared__ int smem_k_cin[BK];
    __shared__ int smem_k_kh [BK];
    __shared__ int smem_k_kw [BK];

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;        // 0..3
    int wx      = warp_id % WARPS_X;        // 0..3

    int block_m_base = blockIdx.x * BM;
    int block_n_base = blockIdx.y * BN;

    int out_HW    = out_H * out_W;
    int kHkW      = kH * kW;
    size_t HW_Cin = (size_t)H_in * W_in * Cin;

    // ---- Precompute M-table once per block ----
    if (tid < BM) {
        int gm = block_m_base + tid;
        if (gm < M) {
            int n_idx = gm / out_HW;
            int hw    = gm % out_HW;
            smem_m_n [tid] = n_idx;
            smem_m_oh[tid] = hw / out_W;
            smem_m_ow[tid] = hw % out_W;
        } else {
            smem_m_n [tid] = -1;
            smem_m_oh[tid] = 0;
            smem_m_ow[tid] = 0;
        }
    }

    // 2x2 accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    int num_tiles = (K_dim + BK - 1) / BK;

    // ---- Macro: refill K-table for tile starting at k_base ----
    #define REFILL_K_TABLE(k_base)                                            \
        if (tid < BK) {                                                       \
            int gk = (k_base) + tid;                                          \
            if (gk < K_dim) {                                                 \
                int cin_idx = gk / kHkW;                                      \
                int kp      = gk % kHkW;                                      \
                smem_k_cin[tid] = cin_idx;                                    \
                smem_k_kh [tid] = kp / kW;                                    \
                smem_k_kw [tid] = kp % kW;                                    \
            } else {                                                          \
                smem_k_cin[tid] = 0; smem_k_kh[tid] = 0; smem_k_kw[tid] = 0;  \
            }                                                                 \
        }

    // ---- Macro: load A tile via per-element implicit GEMM lookup.
    // cp.async can't help here since we need scalar address synthesis per
    // element. We do scalar __half stores into smem with the index logic
    // from v1, but distribute across 512 threads (BM*BK / 512 = 8 elems
    // per thread for BM=128, BK=32). Fully unrolled, no synchronization
    // needed within the loop because each thread writes its own smem slots.
    #define LOAD_A_TILE(buf, k_base)                                          \
        for (int idx = tid; idx < BM * BK; idx += BLOCK_SIZE) {              \
            int srow = idx / BK;                                              \
            int scol = idx % BK;                                              \
            int n_idx   = smem_m_n [srow];                                    \
            int out_h   = smem_m_oh[srow];                                    \
            int out_w   = smem_m_ow[srow];                                    \
            int cin_idx = smem_k_cin[scol];                                   \
            int kh_idx  = smem_k_kh [scol];                                   \
            int kw_idx  = smem_k_kw [scol];                                   \
            __half v = __float2half(0.0f);                                    \
            if (n_idx >= 0 && (k_base) + scol < K_dim) {                      \
                int in_h = out_h + kh_idx - pad;                              \
                int in_w = out_w + kw_idx - pad;                              \
                if ((unsigned)in_h < (unsigned)H_in &&                        \
                    (unsigned)in_w < (unsigned)W_in) {                        \
                    size_t xf = (size_t)n_idx * HW_Cin                        \
                              + (size_t)in_h * W_in * Cin                     \
                              + (size_t)in_w * Cin                            \
                              + cin_idx;                                      \
                    v = X[xf];                                                \
                }                                                             \
            }                                                                 \
            smem_a[buf][srow * STRIDE_A + scol] = v;                          \
        }

    // ---- Macro: load B tile via cp.async (weights are FP16 already) ----
    #define LOAD_B_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_B; _i++) {                            \
            int _flat = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;            \
            int _row  = _flat / BN;                                           \
            int _col  = _flat % BN;                                           \
            int _soff = _row * STRIDE_B + _col;                               \
            int _grow = (k_base) + _row;                                      \
            int _gcol = block_n_base + _col;                                  \
            if (_grow < K_dim && _gcol + ELEMS_PER_COPY - 1 < Cout) {        \
                __pipeline_memcpy_async(                                      \
                    &smem_b[buf][_soff],                                      \
                    &B[(size_t)_grow * Cout + _gcol],                         \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {                 \
                    int _gc = _gcol + _b;                                     \
                    smem_b[buf][_soff + _b] =                                 \
                        (_grow < K_dim && _gc < Cout)                         \
                        ? B[(size_t)_grow * Cout + _gc] : __float2half(0.0f); \
                }                                                             \
            }                                                                 \
        }

    #define COMPUTE_TILE(buf)                                                  \
        _Pragma("unroll")                                                     \
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {              \
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,            \
                           __half, wmma::row_major> a_frag[WARP_TILES_M];     \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++) {                       \
                int a_row = wy * 32 + wi * WMMA_M;                            \
                wmma::load_matrix_sync(a_frag[wi],                            \
                    &smem_a[buf][a_row * STRIDE_A + k_local], STRIDE_A);      \
            }                                                                 \
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,            \
                           __half, wmma::row_major> b_frag[WARP_TILES_N];     \
            _Pragma("unroll")                                                 \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                       \
                int b_col = wx * 32 + wj * WMMA_N;                            \
                wmma::load_matrix_sync(b_frag[wj],                            \
                    &smem_b[buf][k_local * STRIDE_B + b_col], STRIDE_B);      \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++)                         \
                _Pragma("unroll")                                             \
                for (int wj = 0; wj < WARP_TILES_N; wj++)                     \
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj],       \
                                   acc[wi][wj]);                              \
        }

    // ---- Prologue: fill buffer 0 ----
    REFILL_K_TABLE(0);
    __syncthreads();
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // ---- Main loop ----
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        REFILL_K_TABLE(next_k_base);
        __syncthreads();
        LOAD_A_TILE(next_buf, next_k_base);
        LOAD_B_TILE(next_buf, next_k_base);
        __pipeline_commit();
        COMPUTE_TILE(cur_buf);
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ---- Last tile ----
    {
        int last_buf = (num_tiles - 1) & 1;
        COMPUTE_TILE(last_buf);
    }

    #undef REFILL_K_TABLE
    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // ---- Store accumulators to global Y ----
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_m_base + wy * 32 + wi * WMMA_M;
            int c_col = block_n_base + wx * 32 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > Cout) continue;

            wmma::store_matrix_sync(
                &Y[(size_t)c_row * Cout + c_col],
                acc[wi][wj], Cout, wmma::mem_row_major);
        }
    }
}

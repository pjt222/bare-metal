/*
 * conv2d_im2col.cu — Convolution via im2col + WMMA (Tensor Core) GEMM
 *
 * Problem with direct conv2d_nhwc:
 *   The 3×3 kernel reads X nine times from DRAM (once per kernel position).
 *   Effective DRAM traffic = 9 × X_size, not 1×. No shared memory reuse.
 *   Result: ~300 GFLOPS at SD params despite 174 TFLOPS tensor core potential.
 *
 * Solution: im2col + WMMA
 *   1. im2col_nhwc_fp16: reshape X[N,H,W,Cin] → col[M, K] FP16
 *        M = N × out_H × out_W  (one row per output spatial position)
 *        K = Cin × kH × kW      (one col per input patch element)
 *      Each input element is read exactly once. Cost: write 23.6 MB col buffer.
 *
 *   2. wmma_gemm_conv: col[M, K] × W_t[K, Cout] → Y[M, Cout] FP32
 *        Full WMMA (Tensor Core) GEMM using HMMA.16816.F32.
 *        BLOCK_M=64, BLOCK_N=64, BLOCK_K=16. 4 warps × 4 N-tiles = 16 HMMA/K-step.
 *
 * Note: W_t[K, Cout] = W_direct[Cout, kH, kW, Cin] transposed + reshaped.
 *   W_direct[cout, kh, kw, cin] → W_t[cin*kH*kW + kh*kW + kw, cout]
 *   This reshape can be done on the CPU before launching (one-time cost).
 *
 * SASS instructions expected:
 *   im2col_nhwc_fp16:
 *     LDG.E         — global loads of X elements (FP32)
 *     STG.E.128     — vectorized stores to col matrix (FP16)
 *     MUFU.RCP      — division in index decode (compiler may emit)
 *     IMAD          — integer arithmetic for index computation
 *
 *   wmma_gemm_conv:
 *     HMMA.16816.F32 — Tensor Core 16×8×16 FP16 → FP32 matmul
 *     LDS.128        — 128-bit shared memory loads for smem_A / smem_B
 *     STS            — shared memory stores during tile load phase
 *
 * Constraints (required for WMMA alignment):
 *   M = N*out_H*out_W  must be divisible by WMMA_M = 16
 *   K = Cin*kH*kW      must be divisible by WMMA_K = 16
 *   N (Cout)           must be divisible by WMMA_N = 16
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o conv2d_im2col.sm_86.cubin conv2d_im2col.cu
 *   cuobjdump -sass conv2d_im2col.sm_86.cubin | grep HMMA
 *   → HMMA.16816.F32 (QK^T and PV, 16 calls each per warp per K tile)
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// ---- Dimensions ----
#define WARP_SIZE        32
#define GEMM_NUM_WARPS   4
#define GEMM_BLOCK_THREADS (GEMM_NUM_WARPS * WARP_SIZE)  // 128

// ---- WMMA tile size (maps to HMMA.16816) ----
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

// ---- Block-level tile sizes ----
#define GEMM_BLOCK_M   64    // M rows per block = NUM_WARPS * WMMA_M
#define GEMM_BLOCK_N   64    // N cols per block = TILES_N * WMMA_N
#define GEMM_BLOCK_K   16    // K depth per block = WMMA_K (1 mma_sync per K step)
#define GEMM_TILES_N   (GEMM_BLOCK_N / WMMA_N)  // = 4

// ---- Shared memory padding to avoid bank conflicts ----
// Adding 8 half-elements (16 bytes) per row shifts each row to a different bank set.
#define SMEM_A_ROW_STRIDE  (GEMM_BLOCK_K + 8)   // [64 × 24] half = 3 KB
#define SMEM_B_ROW_STRIDE  (GEMM_BLOCK_N + 8)   // [16 × 72] half = 2.25 KB

// Shared memory size (in __half elements):
//   smem_A: GEMM_BLOCK_M × SMEM_A_ROW_STRIDE = 64 × 24 = 1536 halfs = 3 KB
//   smem_B: GEMM_BLOCK_K × SMEM_B_ROW_STRIDE = 16 × 72 = 1152 halfs = 2.25 KB
//   Total: ~5.25 KB — very comfortable
#define SMEM_A_ELEMENTS  (GEMM_BLOCK_M * SMEM_A_ROW_STRIDE)
#define SMEM_B_ELEMENTS  (GEMM_BLOCK_K * SMEM_B_ROW_STRIDE)


// -----------------------------------------------------------------------
// Kernel 1: im2col_nhwc_fp16
//
// Transforms X[N, H, W, Cin] (FP32, NHWC layout) into the column matrix
// col[M, K] (FP16), where:
//   M = N × out_H × out_W   (one row per output spatial position)
//   K = Cin × kH × kW       (one column per input patch element)
//
// col[m, k] = X[n, out_h + kh - pad, out_w + kw - pad, cin]
//   where m encodes (n, out_h, out_w) and k encodes (cin, kh, kw).
//   Zero for out-of-bounds positions (zero-padding).
//
// Grid:  (ceil(M×K / 256), 1, 1)
// Block: (256, 1, 1)
//
// Adjacent threads differ by 1 in k (col_col) → adjacent writes to col →
// coalesced 128-bit store (compiler may vectorize 8 consecutive halfs).
// -----------------------------------------------------------------------
extern "C" __global__
void im2col_nhwc_fp16(
    const float * __restrict__ X,    // [N, H, W, Cin] FP32 input (NHWC)
    __half       * __restrict__ col, // [M, K] FP16 output
    int N, int H, int W_dim, int Cin,
    int kH, int kW, int pad,
    int out_H, int out_W
) {
    int M_dim = N * out_H * out_W;
    int K_dim = Cin * kH * kW;

    size_t total_elements = (size_t)M_dim * K_dim;

    // Grid-stride loop: each thread processes multiple (col_row, col_col) pairs
    for (size_t flat_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         flat_idx < total_elements;
         flat_idx += (size_t)gridDim.x * blockDim.x)
    {
        int col_row = (int)(flat_idx / K_dim);   // output spatial position
        int col_col = (int)(flat_idx % K_dim);   // input patch element

        // Decode col_row → (n, out_h, out_w)
        int n_idx  = col_row / (out_H * out_W);
        int hw     = col_row % (out_H * out_W);
        int out_h  = hw / out_W;
        int out_w  = hw % out_W;

        // Decode col_col → (cin, kh, kw)
        // Ordering: k = cin * kH * kW + kh * kW + kw
        int cin    = col_col / (kH * kW);
        int k_pos  = col_col % (kH * kW);
        int kh_idx = k_pos / kW;
        int kw_idx = k_pos % kW;

        // Input coordinates (same-padding: stride=1, pad=(kH-1)/2)
        int in_h = out_h + kh_idx - pad;
        int in_w = out_w + kw_idx - pad;

        // Zero-pad: use unsigned comparison to check [0, H) and [0, W) in one op
        float val = 0.0f;
        if ((unsigned)in_h < (unsigned)H && (unsigned)in_w < (unsigned)W_dim) {
            // NHWC flat index: X[n, in_h, in_w, cin]
            size_t x_flat = ((size_t)n_idx * H * W_dim + in_h * W_dim + in_w) * Cin + cin;
            val = X[x_flat];
        }

        col[flat_idx] = __float2half(val);
    }
}


// -----------------------------------------------------------------------
// Kernel 2: wmma_gemm_conv
//
// Computes C = A × B using WMMA Tensor Cores (HMMA.16816.F32):
//   A: col[M, K]   FP16, row-major  (im2col output)
//   B: W_t[K, N]   FP16, row-major  (weights, reshaped: [Cin*kH*kW, Cout])
//   C: Y[M, N]     FP32              (convolution output, flattened: [N*H*W, Cout])
//
// Block tile: [BLOCK_M=64] × [BLOCK_N=64], K-loop step: BLOCK_K=16
//
// Warp assignment (4 warps × 128 threads):
//   Warp i → rows [i×16 : (i+1)×16] of the block's M-tile (one WMMA_M slice)
//   All 4 warps → all 4 N-tiles of width 16 (the full 64-col N-tile)
//   → Each warp accumulates 4 output fragments: c_frag[0..3]
//
// Shared memory (smem_A + smem_B, padded to avoid bank conflicts):
//   smem_A [64 × 24] FP16 = 3 KB   (A tile + 8-col padding per row)
//   smem_B [16 × 72] FP16 = 2.25 KB (B tile + 8-col padding per row)
//   Total: ~5.25 KB — far below 48 KB limit; easy to double-buffer if needed
//
// Grid:  (ceil(M / BLOCK_M), ceil(N / BLOCK_N), 1)
// Block: (128, 1, 1) = 4 warps
//
// Requirements: M, K, N must be divisible by 16 (WMMA dimension constraint).
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(GEMM_BLOCK_THREADS)
void wmma_gemm_conv(
    const __half * __restrict__ A,  // [M, K] FP16 im2col matrix
    const __half * __restrict__ B,  // [K, N] FP16 weight matrix (pre-reshaped)
    float        * __restrict__ C,  // [M, N] FP32 output
    int M, int K_dim, int N
) {
    // Shared memory: smem_A then smem_B, packed tightly
    extern __shared__ __half smem_raw[];
    __half *smem_A = smem_raw;
    __half *smem_B = smem_raw + SMEM_A_ELEMENTS;

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;

    // Block-level tile base indices in the full M×N output matrix
    int block_m_base = blockIdx.x * GEMM_BLOCK_M;
    int block_n_base = blockIdx.y * GEMM_BLOCK_N;

    // Warp's row offset within the block (each warp handles 16 M-rows = WMMA_M)
    int warp_m_base = warp_id * WMMA_M;

    // Initialize accumulator fragments for all 4 N-tiles (one per warp)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[GEMM_TILES_N];
    #pragma unroll
    for (int n_tile = 0; n_tile < GEMM_TILES_N; n_tile++) {
        wmma::fill_fragment(c_frag[n_tile], 0.0f);
    }

    // ================================================================
    // Main K-tile loop: iterate over K in steps of BLOCK_K=16
    // ================================================================
    for (int k_base = 0; k_base < K_dim; k_base += GEMM_BLOCK_K) {

        // ---- Load smem_A: [BLOCK_M × BLOCK_K] from A[block_m_base:+64, k_base:+16] ----
        // 64 × 16 = 1024 elements; 128 threads → 8 elements each.
        // Linear index: thread t loads element t, t+128, ..., t+7×128.
        // Row = idx / BLOCK_K, Col = idx % BLOCK_K → coalesced (adjacent idx → adjacent col).
        for (int load_idx = global_thread;
             load_idx < GEMM_BLOCK_M * GEMM_BLOCK_K;
             load_idx += GEMM_BLOCK_THREADS)
        {
            int smem_row = load_idx / GEMM_BLOCK_K;
            int smem_col = load_idx % GEMM_BLOCK_K;
            int global_m = block_m_base + smem_row;
            int global_k = k_base       + smem_col;

            __half val = (global_m < M && global_k < K_dim)
                         ? A[(size_t)global_m * K_dim + global_k]
                         : __float2half(0.0f);

            // Store with row padding: stride = SMEM_A_ROW_STRIDE = BLOCK_K + 8
            smem_A[smem_row * SMEM_A_ROW_STRIDE + smem_col] = val;
        }

        // ---- Load smem_B: [BLOCK_K × BLOCK_N] from B[k_base:+16, block_n_base:+64] ----
        // 16 × 64 = 1024 elements; same pattern as above.
        // Row = idx / BLOCK_N → K dimension; Col = idx % BLOCK_N → N dimension.
        // Adjacent idx (same row) differ in col → adjacent N → coalesced global reads.
        for (int load_idx = global_thread;
             load_idx < GEMM_BLOCK_K * GEMM_BLOCK_N;
             load_idx += GEMM_BLOCK_THREADS)
        {
            int smem_row = load_idx / GEMM_BLOCK_N;
            int smem_col = load_idx % GEMM_BLOCK_N;
            int global_k = k_base       + smem_row;
            int global_n = block_n_base + smem_col;

            __half val = (global_k < K_dim && global_n < N)
                         ? B[(size_t)global_k * N + global_n]
                         : __float2half(0.0f);

            smem_B[smem_row * SMEM_B_ROW_STRIDE + smem_col] = val;
        }

        __syncthreads();

        // ---- WMMA computation: 1 a_frag × 4 b_frags per warp ----
        // Since BLOCK_K = WMMA_K = 16, there is exactly 1 mma_sync step per K tile.
        // Each warp loads its 16-row slice of smem_A and all 4 N-tile slices of smem_B.
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;

            // A fragment: warp's 16 rows, all K=16 cols
            wmma::load_matrix_sync(a_frag,
                smem_A + warp_m_base * SMEM_A_ROW_STRIDE,
                SMEM_A_ROW_STRIDE);

            #pragma unroll
            for (int n_tile = 0; n_tile < GEMM_TILES_N; n_tile++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;

                // B fragment: all K=16 rows, this N-tile's 16 cols
                wmma::load_matrix_sync(b_frag,
                    smem_B + n_tile * WMMA_N,   // col offset: n_tile × 16
                    SMEM_B_ROW_STRIDE);

                wmma::mma_sync(c_frag[n_tile], a_frag, b_frag, c_frag[n_tile]);
            }
        }

        __syncthreads();
    }
    // ================================================================
    // End K-tile loop
    // ================================================================

    // ---- Store accumulator fragments to global C ----
    // Each warp stores 4 fragments covering [warp_m_base:+16, 0:64] within the block tile.
    int global_m_warp = block_m_base + warp_m_base;
    if (global_m_warp < M) {
        #pragma unroll
        for (int n_tile = 0; n_tile < GEMM_TILES_N; n_tile++) {
            int global_n_tile = block_n_base + n_tile * WMMA_N;
            if (global_n_tile < N) {
                wmma::store_matrix_sync(
                    C + (size_t)global_m_warp * N + global_n_tile,
                    c_frag[n_tile],
                    N,
                    wmma::mem_row_major);
            }
        }
    }
}

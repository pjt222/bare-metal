/*
 * conv2d_implicit_gemm.cu — Implicit GEMM for Conv2d (no col buffer)
 *
 * Problem with explicit im2col + WMMA GEMM:
 *   1. Writes col buffer [M, K] FP16 = [N*OH*OW, Cin*kH*kW] = 23.6 MB at SD params
 *   2. col is written (1 DRAM pass) then read back (1 DRAM pass) → 47.2 MB extra BW
 *   3. col exceeds L2 (4 MB) so it can never be cached between passes
 *
 * Solution: compute im2col indices on-the-fly inside WMMA tile loading.
 *
 * Key optimization — precomputed coordinate tables (critical for performance):
 *   Naively decoding (global_m, global_k) per element requires 6 integer divisions
 *   per A-tile element × 1024 elements per tile = 6144 integer divisions per K-tile.
 *   Integer division is ~20 cycles on Ampere → ~123K cycles of decode overhead.
 *
 *   With precomputed tables (stored in shared memory):
 *   - smem_m_n[64]:    n_idx  for each M-row in this block tile  (constant across K)
 *   - smem_m_oh[64]:   out_h  for each M-row                     (constant across K)
 *   - smem_m_ow[64]:   out_w  for each M-row                     (constant across K)
 *   - smem_k_cin[16]:  cin    for each K-col in this K-tile      (recomputed each K-iter)
 *   - smem_k_kh[16]:   kh     for each K-col                     (recomputed each K-iter)
 *   - smem_k_kw[16]:   kw     for each K-col                     (recomputed each K-iter)
 *
 *   Per-element A load reduces to:
 *     in_h = smem_m_oh[smem_row] + smem_k_kh[smem_col] - pad
 *     in_w = smem_m_ow[smem_row] + smem_k_kw[smem_col] - pad
 *     X[smem_m_n[smem_row] * H*W + in_h * W + in_w] * Cin + smem_k_cin[smem_col]
 *   → Only adds/compares per element; divisions amortized across 64 M-rows and 16 K-cols.
 *
 * Total smem per block:
 *   smem_A:   BLOCK_M × (BLOCK_K+8) halfs = 64×24 = 3 KB
 *   smem_B:   BLOCK_K × (BLOCK_N+8) halfs = 16×72 = 2.25 KB
 *   m_tables: 3 × BLOCK_M ints = 3×64×4 = 768 bytes = 0.75 KB
 *   k_tables: 3 × BLOCK_K ints = 3×16×4 = 192 bytes = 0.19 KB
 *   Total: ~6.2 KB — well within 48 KB limit
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 \
 *        -o conv2d_implicit_gemm.sm_86.cubin conv2d_implicit_gemm.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// ---- Thread/Warp constants ----
#define WARP_SIZE        32
#define NUM_WARPS        4
#define BLOCK_THREADS    (NUM_WARPS * WARP_SIZE)   // 128

// ---- WMMA tile dimensions ----
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

// ---- Block-level tile sizes (identical to conv2d_im2col) ----
#define BLOCK_M     64    // M rows per block (4 warps × WMMA_M)
#define BLOCK_N     64    // N cols per block (4 N-tiles × WMMA_N)
#define BLOCK_K     16    // K depth per block = WMMA_K
#define TILES_N     (BLOCK_N / WMMA_N)   // = 4

// ---- Shared memory strides (padded by 8 halfs to avoid bank conflicts) ----
#define SMEM_A_STRIDE  (BLOCK_K + 8)    // 24 halfs per row → 64×24 = 1536 halfs = 3 KB
#define SMEM_B_STRIDE  (BLOCK_N + 8)    // 72 halfs per row → 16×72 = 1152 halfs = 2.25 KB
#define SMEM_A_ELEMS   (BLOCK_M * SMEM_A_STRIDE)
#define SMEM_B_ELEMS   (BLOCK_K * SMEM_B_STRIDE)

// ---- Coordinate table sizes (precomputed per block / per K-tile) ----
// 3 arrays × BLOCK_M ints for M-dim: n_idx, out_h, out_w
// 3 arrays × BLOCK_K ints for K-dim: cin, kh, kw
#define M_TABLE_ELEMS  (3 * BLOCK_M)   // 192 ints = 768 bytes
#define K_TABLE_ELEMS  (3 * BLOCK_K)   // 48 ints = 192 bytes

// Shared memory layout (in units of int32, after casting):
//   [SMEM_A halfs | SMEM_B halfs | M_TABLE ints | K_TABLE ints]
//   (halfs and ints are interleaved safely using a union or careful byte offsets)

// Total smem bytes:
//   SMEM_A: 3072 B, SMEM_B: 2304 B, M_TABLE: 768 B, K_TABLE: 192 B = 6336 bytes ≈ 6.2 KB


extern "C" __global__ __launch_bounds__(BLOCK_THREADS)
void implicit_gemm_conv(
    const float * __restrict__ X,    // [N, H_in, W_in, Cin]  FP32 NHWC input
    const __half * __restrict__ B,   // [K_dim, Cout] FP16 weights (pre-reshaped)
    float        * __restrict__ Y,   // [M, Cout] FP32 output = [N*out_H*out_W, Cout]
    int N_batch, int H_in, int W_in, int Cin,
    int kH, int kW, int pad,
    int out_H, int out_W,
    int M,       // = N_batch × out_H × out_W
    int K_dim,   // = Cin × kH × kW
    int Cout     // number of output channels
) {
    // ---- Shared memory partitioning ----
    // Use raw byte array to overlay half and int arrays cleanly.
    extern __shared__ char smem_bytes[];

    __half *smem_A   = (__half *)(smem_bytes);
    __half *smem_B   = (__half *)(smem_bytes + SMEM_A_ELEMS * sizeof(__half));

    // Coordinate tables immediately after WMMA tiles
    int smem_half_bytes = (SMEM_A_ELEMS + SMEM_B_ELEMS) * sizeof(__half);
    // Align to 4-byte boundary (already aligned since SMEM_A_ELEMS+SMEM_B_ELEMS is even)
    int *smem_m_n    = (int *)(smem_bytes + smem_half_bytes);               // n_idx per M-row
    int *smem_m_oh   = smem_m_n  + BLOCK_M;                                 // out_h per M-row
    int *smem_m_ow   = smem_m_oh + BLOCK_M;                                 // out_w per M-row
    int *smem_k_cin  = smem_m_ow + BLOCK_M;                                 // cin per K-col
    int *smem_k_kh   = smem_k_cin + BLOCK_K;                                // kh per K-col
    int *smem_k_kw   = smem_k_kh  + BLOCK_K;                                // kw per K-col

    int thread_id  = threadIdx.x;
    int warp_id    = thread_id / WARP_SIZE;

    // Block-level output tile base
    int block_m_base = blockIdx.x * BLOCK_M;
    int block_n_base = blockIdx.y * BLOCK_N;

    // Each warp covers its 16-row slice of M
    int warp_m_base = warp_id * WMMA_M;

    // ---- Precompute M-dimension coordinate table (constant for entire block) ----
    // Use first 64 threads: thread t decodes global_m = block_m_base + t.
    // This amortizes ~4 integer divisions × 64 rows over all K-tile iterations.
    int out_HW     = out_H * out_W;
    int kH_kW_prod = kH * kW;

    if (thread_id < BLOCK_M) {
        int global_m = block_m_base + thread_id;
        if (global_m < M) {
            int n_idx  = global_m / out_HW;
            int hw     = global_m % out_HW;
            int out_h  = hw / out_W;
            int out_w  = hw % out_W;
            smem_m_n [thread_id] = n_idx;
            smem_m_oh[thread_id] = out_h;
            smem_m_ow[thread_id] = out_w;
        } else {
            // Out-of-bounds M rows: mark with sentinel so A-tile loading emits 0
            smem_m_n [thread_id] = -1;
            smem_m_oh[thread_id] = -1;
            smem_m_ow[thread_id] = -1;
        }
    }
    // K-dimension tables will be filled inside the K-loop (change each iteration)

    // Initialize accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[TILES_N];
    #pragma unroll
    for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
        wmma::fill_fragment(c_frag[n_tile], 0.0f);
    }

    __syncthreads();   // ensure M-table is visible to all threads

    // ================================================================
    // K-tile loop
    // ================================================================
    for (int k_base = 0; k_base < K_dim; k_base += BLOCK_K) {

        // ---- Recompute K-dimension coordinate table for this K-tile ----
        // Use first 16 threads: thread t decodes global_k = k_base + t.
        // Cost: 4 divs × 16 threads = 64 integer divisions per K-tile (vs 6144 naively).
        if (thread_id < BLOCK_K) {
            int global_k = k_base + thread_id;
            if (global_k < K_dim) {
                int cin_idx = global_k / kH_kW_prod;
                int k_pos   = global_k % kH_kW_prod;
                int kh_idx  = k_pos / kW;
                int kw_idx  = k_pos % kW;
                smem_k_cin[thread_id] = cin_idx;
                smem_k_kh [thread_id] = kh_idx;
                smem_k_kw [thread_id] = kw_idx;
            } else {
                smem_k_cin[thread_id] = 0;
                smem_k_kh [thread_id] = 0;
                smem_k_kw [thread_id] = 0;
            }
        }

        __syncthreads();  // K-table must be ready before A-tile loading

        // ---- Load smem_A: 64×16 elements using precomputed coordinate tables ----
        // Per element: just 2 adds + 2 comparisons + 1 array lookup → no divisions.
        for (int load_idx = thread_id;
             load_idx < BLOCK_M * BLOCK_K;
             load_idx += BLOCK_THREADS)
        {
            int smem_row = load_idx / BLOCK_K;  // 0..63 (M dimension)
            int smem_col = load_idx % BLOCK_K;  // 0..15 (K dimension)

            // Fetch precomputed coordinates from shared memory
            int n_idx   = smem_m_n [smem_row];
            int out_h   = smem_m_oh[smem_row];
            int out_w   = smem_m_ow[smem_row];
            int cin_idx = smem_k_cin[smem_col];
            int kh_idx  = smem_k_kh [smem_col];
            int kw_idx  = smem_k_kw [smem_col];

            __half val = __float2half(0.0f);

            // n_idx == -1 signals out-of-bounds M row (emit zero for all K)
            if (n_idx >= 0 && (k_base + smem_col) < K_dim) {
                int in_h = out_h + kh_idx - pad;
                int in_w = out_w + kw_idx - pad;

                // Bounds check (unsigned comparison handles negative via wraparound)
                if ((unsigned)in_h < (unsigned)H_in &&
                    (unsigned)in_w < (unsigned)W_in)
                {
                    size_t x_flat = ((size_t)n_idx * H_in * W_in
                                    + in_h * W_in
                                    + in_w) * Cin + cin_idx;
                    val = __float2half(X[x_flat]);
                }
            }

            smem_A[smem_row * SMEM_A_STRIDE + smem_col] = val;
        }

        // ---- Load smem_B: 16×64 elements — same as wmma_gemm_conv ----
        for (int load_idx = thread_id;
             load_idx < BLOCK_K * BLOCK_N;
             load_idx += BLOCK_THREADS)
        {
            int smem_row = load_idx / BLOCK_N;
            int smem_col = load_idx % BLOCK_N;
            int global_k = k_base       + smem_row;
            int global_n = block_n_base + smem_col;

            __half val = (global_k < K_dim && global_n < Cout)
                         ? B[(size_t)global_k * Cout + global_n]
                         : __float2half(0.0f);

            smem_B[smem_row * SMEM_B_STRIDE + smem_col] = val;
        }

        __syncthreads();

        // ---- WMMA computation — IDENTICAL to wmma_gemm_conv ----
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;

            wmma::load_matrix_sync(a_frag,
                smem_A + warp_m_base * SMEM_A_STRIDE,
                SMEM_A_STRIDE);

            #pragma unroll
            for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;

                wmma::load_matrix_sync(b_frag,
                    smem_B + n_tile * WMMA_N,
                    SMEM_B_STRIDE);

                wmma::mma_sync(c_frag[n_tile], a_frag, b_frag, c_frag[n_tile]);
            }
        }

        __syncthreads();
    }

    // ---- Store accumulator fragments to global Y ----
    int global_m_warp = block_m_base + warp_m_base;
    if (global_m_warp < M) {
        #pragma unroll
        for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
            int global_n_tile = block_n_base + n_tile * WMMA_N;
            if (global_n_tile < Cout) {
                wmma::store_matrix_sync(
                    Y + (size_t)global_m_warp * Cout + global_n_tile,
                    c_frag[n_tile],
                    Cout,
                    wmma::mem_row_major);
            }
        }
    }
}

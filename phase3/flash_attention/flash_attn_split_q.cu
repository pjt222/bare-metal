/*
 * flash_attn_split_q.cu — Split-Q Flash Attention (two-kernel pipeline)
 *
 * Restructures the Flash Attention grid so each block handles a SUBSET of KV
 * tiles and iterates over ALL Q tiles, enabling K/V reuse across Q blocks.
 *
 * Current (br16):   Grid=(q_tiles, heads, batch) — each block reads ALL KV tiles.
 *   Problem: at seq=1024, K/V loaded 16× from DRAM (L2 evicted between Q-block waves).
 *
 * Split-Q:          Grid=(num_splits, heads, batch) — each block reads a KV CHUNK.
 *   Benefit: K/V DRAM traffic reduced by up to num_splits×. Q stays hot in L2
 *            because all blocks access Q tiles in the same order.
 *   Cost:    requires a second reduction kernel to merge partial softmax results.
 *
 * Kernel 1 (flash_attn_split_q):
 *   For each Q tile [q_base, q_base+64):
 *     1. Reset running_max, running_sum, smem_pv
 *     2. For each KV tile in assigned range [kv_start, kv_end):
 *          Same inner loop as br16: load K/V, QK^T WMMA, online softmax, PV WMMA
 *     3. Store partial {m, l, O_unnorm} for the 64 Q rows
 *
 * Kernel 2 (flash_attn_split_q_reduce):
 *   For each Q row: merge partial {m, l, O} across splits using log-sum-exp recombination,
 *   then normalize to produce final output.
 *
 * Partial output layout:
 *   partial_m: [num_splits × batch × heads × seq_len]           float
 *   partial_l: [num_splits × batch × heads × seq_len]           float
 *   partial_O: [num_splits × batch × heads × seq_len × D_HEAD]  float
 *
 * SASS instructions (same as br16):
 *   HMMA.16816.F32, SHFL.BFLY, MUFU.EX2, MUFU.RCP, LDS, STS
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_split_q.sm_86.cubin flash_attn_split_q.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16                         // query rows per warp (= WMMA_M)
#define Br_BLOCK    (NUM_WARPS * Br_WARP)      // = 64 query rows per block
#define Bc          64                         // KV tile size
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)          // = 4
#define TILES_Bc    (Bc / WMMA_N)              // = 4

#define LOG2E   1.4426950408889634f

// -----------------------------------------------------------------------
// Kernel 1: flash_attn_split_q
//
// Grid:  (num_splits, num_heads, batch_size)
// Block: (128, 1, 1) = 4 warps
// Smem:  48 KB (same layout as br16)
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_split_q(
    const __half * __restrict__ Q,           // [batch × heads × seq_len × D_HEAD] FP16
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ partial_O,   // [num_splits × batch × heads × seq_len × D_HEAD]
    float        * __restrict__ partial_m,   // [num_splits × batch × heads × seq_len]
    float        * __restrict__ partial_l,   // [num_splits × batch × heads × seq_len]
    int   seq_len,
    int   num_heads,
    int   num_splits,
    float scale
) {
    // ---- Shared memory (48 KB, same as br16) ----
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));
    float  *smem_pv   = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half)
                                           + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    int split_idx  = blockIdx.x;
    int head_idx   = blockIdx.y;
    int batch_idx  = blockIdx.z;
    int batch_size = gridDim.z;

    // ---- Compute this split's KV tile range ----
    int num_kv_tiles    = (seq_len + Bc - 1) / Bc;
    int tiles_per_split = (num_kv_tiles + num_splits - 1) / num_splits;
    int kv_tile_start   = split_idx * tiles_per_split;
    int kv_tile_end     = kv_tile_start + tiles_per_split;
    if (kv_tile_end > num_kv_tiles) kv_tile_end = num_kv_tiles;
    int kv_start = kv_tile_start * Bc;
    int kv_end   = kv_tile_end * Bc;
    if (kv_end > seq_len) kv_end = seq_len;

    // ---- Q/K/V head pointers (same as br16) ----
    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;
    size_t base_offset  = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;

    const __half *Q_head = Q + base_offset;
    const __half *K_head = K + base_offset;
    const __half *V_head = V + base_offset;

    // ---- Partial output offset (flat indexing into [split, batch, head, seq, ...]) ----
    size_t ml_split_stride = (size_t)batch_size * num_heads * seq_len;
    size_t ml_base = (size_t)split_idx * ml_split_stride
                   + (size_t)batch_idx * num_heads * seq_len
                   + (size_t)head_idx  * seq_len;

    size_t o_split_stride = ml_split_stride * D_HEAD;
    size_t o_base  = (size_t)split_idx * o_split_stride
                   + (size_t)batch_idx * num_heads * seq_len * D_HEAD
                   + (size_t)head_idx  * seq_len * D_HEAD;

    int num_q_tiles = (seq_len + Br_BLOCK - 1) / Br_BLOCK;

    // Per-warp smem section pointers (same as br16 — warp_id is constant)
    float *warp_work = smem_work + warp_id * Br_WARP * Bc;
    float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;

    // ====================================================================
    // Outer loop: iterate over ALL Q tiles
    // ====================================================================
    for (int qt = 0; qt < num_q_tiles; qt++) {
        int block_q_base = qt * Br_BLOCK;
        int warp_q_base  = block_q_base + warp_id * Br_WARP;
        bool valid_warp  = (warp_q_base < seq_len);

        // ---- Initialize smem_pv to 0 (all 128 threads) ----
        for (int idx = global_thread; idx < Br_BLOCK * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
            smem_pv[idx] = 0.0f;
        }

        // ---- Initialize per-warp running online softmax state ----
        float running_max[Br_WARP];
        float running_sum[Br_WARP];
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            running_max[row] = -3.402823466e+38f;
            running_sum[row] = 0.0f;
        }

        __syncthreads();

        // ================================================================
        // Inner loop: iterate over this split's KV tiles
        // ================================================================
        for (int kv_base = kv_start; kv_base < kv_end; kv_base += Bc) {

            // Phase A: Load K_tile + V_tile (all 128 threads, coalesced)
            for (int idx = global_thread; idx < Bc * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
                int kv_row    = idx / D_HEAD;
                int d_col     = idx % D_HEAD;
                int kv_global = kv_base + kv_row;
                K_tile[idx] = (kv_global < seq_len)
                    ? K_head[(size_t)kv_global * D_HEAD + d_col]
                    : __float2half(0.0f);
                V_tile[idx] = (kv_global < seq_len)
                    ? V_head[(size_t)kv_global * D_HEAD + d_col]
                    : __float2half(0.0f);
            }
            __syncthreads();

            // Phase B: QK^T via WMMA → score_frag → warp_work (FP32)
            if (valid_warp) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag[TILES_Bc];
                #pragma unroll
                for (int n = 0; n < TILES_Bc; n++) wmma::fill_fragment(score_frag[n], 0.0f);

                #pragma unroll
                for (int dk = 0; dk < TILES_D; dk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
                    wmma::load_matrix_sync(q_frag,
                        Q_head + (size_t)warp_q_base * D_HEAD + dk * WMMA_K,
                        D_HEAD);

                    #pragma unroll
                    for (int n = 0; n < TILES_Bc; n++) {
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                        wmma::load_matrix_sync(k_frag,
                            K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                            D_HEAD);
                        wmma::mma_sync(score_frag[n], q_frag, k_frag, score_frag[n]);
                    }
                }

                // Scale scores and store to warp_work [16 × Bc]
                #pragma unroll
                for (int n = 0; n < TILES_Bc; n++) {
                    #pragma unroll
                    for (int elem_idx = 0; elem_idx < score_frag[n].num_elements; elem_idx++) {
                        score_frag[n].x[elem_idx] *= scale;
                    }
                    wmma::store_matrix_sync(
                        warp_work + n * WMMA_N,
                        score_frag[n],
                        Bc,
                        wmma::mem_row_major);
                }
            }

            __syncthreads();

            // Phase C: Online softmax update
            if (valid_warp) {
                #pragma unroll
                for (int row = 0; row < Br_WARP; row++) {
                    float *score_row = warp_work + row * Bc;
                    float *pv_row    = warp_pv   + row * D_HEAD;

                    // Max reduction across Bc=64 scores
                    float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                    #pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                    float new_max        = fmaxf(running_max[row], partial_max);
                    float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                    running_max[row]     = new_max;

                    // Rescale running output AND running sum
                    pv_row[lane]             *= rescale_factor;
                    pv_row[lane + WARP_SIZE] *= rescale_factor;
                    running_sum[row]         *= rescale_factor;  // correct recurrence: l = l*rescale + new_sum

                    // Compute exp weights
                    float w_lo = exp2f((score_row[lane]             - new_max) * LOG2E);
                    float w_hi = exp2f((score_row[lane + WARP_SIZE] - new_max) * LOG2E);

                    // Store FP16 weights (overlay onto score region)
                    __half *weight_row = (__half*)warp_work + row * Bc;
                    weight_row[lane]             = __float2half(w_lo);
                    weight_row[lane + WARP_SIZE] = __float2half(w_hi);

                    // Running sum: accumulate exp weight total
                    float partial_sum = w_lo + w_hi;
                    #pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                    running_sum[row] += partial_sum;
                }

                // Phase D: PV WMMA — weight_frag[16×Bc] × V_tile[Bc×D_HEAD] → smem_pv
                __half *weight_ptr = (__half*)warp_work;

                #pragma unroll
                for (int n = 0; n < TILES_D; n++) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum;
                    wmma::load_matrix_sync(
                        pv_accum,
                        warp_pv + n * WMMA_N,
                        D_HEAD,
                        wmma::mem_row_major);

                    #pragma unroll
                    for (int k = 0; k < TILES_Bc; k++) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                        wmma::load_matrix_sync(w_frag,
                            weight_ptr + k * WMMA_K,
                            Bc);

                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                        wmma::load_matrix_sync(v_frag,
                            V_tile + k * WMMA_K * D_HEAD + n * WMMA_N,
                            D_HEAD);

                        wmma::mma_sync(pv_accum, w_frag, v_frag, pv_accum);
                    }

                    wmma::store_matrix_sync(
                        warp_pv + n * WMMA_N,
                        pv_accum,
                        D_HEAD,
                        wmma::mem_row_major);
                }
            }

            __syncthreads();  // all warps done — smem safe for next KV tile
        }
        // ================================================================
        // End of KV loop — write partial results for this Q tile
        // ================================================================
        if (valid_warp) {
            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                int global_q = warp_q_base + row;
                if (global_q >= seq_len) break;

                partial_m[ml_base + global_q] = running_max[row];
                partial_l[ml_base + global_q] = running_sum[row];

                float *pv_row = warp_pv + row * D_HEAD;
                size_t o_idx  = o_base + (size_t)global_q * D_HEAD;
                partial_O[o_idx + lane]             = pv_row[lane];
                partial_O[o_idx + lane + WARP_SIZE] = pv_row[lane + WARP_SIZE];
            }
        }

        __syncthreads();  // ensure all warps done before next Q tile zeros smem_pv
    }
}

// -----------------------------------------------------------------------
// Kernel 2: flash_attn_split_q_reduce
//
// Merges partial {m, l, O_unnorm} across splits via log-sum-exp recombination.
//
// For each Q row:
//   m_global = max over splits of partial_m[s]
//   l_total  = sum over splits of partial_l[s] * exp(partial_m[s] - m_global)
//   o_total  = sum over splits of partial_O[s] * exp(partial_m[s] - m_global)
//   O_final  = o_total / l_total
//
// Grid:  (ceil(seq_len / Br_BLOCK), num_heads, batch_size)
// Block: (128, 1, 1) = 4 warps, each handling 16 Q rows
// No shared memory needed.
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_split_q_reduce(
    const float * __restrict__ partial_O,   // [num_splits × batch × heads × seq_len × D_HEAD]
    const float * __restrict__ partial_m,   // [num_splits × batch × heads × seq_len]
    const float * __restrict__ partial_l,   // [num_splits × batch × heads × seq_len]
    float       * __restrict__ O,           // [batch × heads × seq_len × D_HEAD]
    int   seq_len,
    int   num_heads,
    int   num_splits
) {
    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    int block_q_base = blockIdx.x * Br_BLOCK;
    int warp_q_base  = block_q_base + warp_id * Br_WARP;
    int head_idx     = blockIdx.y;
    int batch_idx    = blockIdx.z;
    int batch_size   = gridDim.z;

    if (warp_q_base >= seq_len) return;

    // Output pointer
    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;
    size_t out_base     = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;

    // Partial array strides
    size_t ml_split_stride = (size_t)batch_size * num_heads * seq_len;
    size_t o_split_stride  = ml_split_stride * D_HEAD;
    size_t ml_row_base     = (size_t)batch_idx * num_heads * seq_len
                           + (size_t)head_idx * seq_len;
    size_t o_row_base      = ml_row_base * D_HEAD;

    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        int global_q = warp_q_base + row;
        if (global_q >= seq_len) break;

        // ---- Pass 1: find global max across all splits ----
        float m_global = -3.402823466e+38f;
        for (int s = 0; s < num_splits; s++) {
            float ms = partial_m[s * ml_split_stride + ml_row_base + global_q];
            m_global = fmaxf(m_global, ms);
        }

        // ---- Pass 2: accumulate corrected l and O ----
        float l_total = 0.0f;
        float o_lo    = 0.0f;
        float o_hi    = 0.0f;

        for (int s = 0; s < num_splits; s++) {
            float ms   = partial_m[s * ml_split_stride + ml_row_base + global_q];
            float ls   = partial_l[s * ml_split_stride + ml_row_base + global_q];
            float corr = exp2f((ms - m_global) * LOG2E);

            l_total += ls * corr;

            size_t o_idx = s * o_split_stride + o_row_base + (size_t)global_q * D_HEAD;
            o_lo += partial_O[o_idx + lane]             * corr;
            o_hi += partial_O[o_idx + lane + WARP_SIZE] * corr;
        }

        // ---- Normalize and store ----
        float rcp_l = __frcp_rn(l_total);
        size_t out_idx = out_base + (size_t)global_q * D_HEAD;
        O[out_idx + lane]             = o_lo * rcp_l;
        O[out_idx + lane + WARP_SIZE] = o_hi * rcp_l;
    }
}

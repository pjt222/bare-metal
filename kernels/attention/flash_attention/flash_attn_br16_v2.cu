/*
 * flash_attn_br16_v2.cu
 *   Flash Attention Br=16 — CANONICAL high-performance kernel.
 *
 * Builds on three stages of optimization (issue #29):
 *   1. Lean per-thread softmax state  (was: flash_attn_br16_regpv_lean.cu)
 *   2. Q register cache               (was: flash_attn_br16_regpv_lean_qcache.cu)
 *   3. smem_work elimination via fragment-shfl reductions
 *
 * See `docs/fragment_shfl_reductions.md` for the reusable pattern.
 * See `docs/gpu_reflections.md` Observation P for the optimization story.
 *
 * Performance (RTX 3070 Ti, sm_86):
 *   seq=1024 b=8 h=8: 1.75 ms, 10000 GFLOPS  (1.40× vs flash_attn_br16_regpv)
 *   ~10 TFLOPS plateau across seq 512..4096 (5.7% of FP16 TC peak 174 TFLOPS)
 *
 * Original verbose-name description follows for documentation continuity:
 *
 * Builds on flash_attn_br16_regpv_lean_qcache.cu by:
 *   1. Keeping FP32 score_frag in registers across Phase B → Phase C
 *      (skips 16 KB of FP32 smem_work + 16 wmma::store_matrix_sync calls)
 *   2. Per-row max/sum reductions performed directly on score_frag elements
 *      via fragment-element local fmax + intra-group SHFL.BFLY (lanes 0-3
 *      in each thread group hold the same row's 4 fragment elements per tile)
 *   3. FP16 weights written directly to a small 8 KB smem region for Phase D
 *      load_matrix_sync (Phase D unchanged structurally)
 *   4. Final output: pv_accum elements scaled by 1/sum and written directly
 *      to global O (no smem staging)
 *
 * smem layout (24 KB total, 8 KB less than baseline):
 *   K_tile     [Bc × D_HEAD] FP16 = 8 KB
 *   V_tile     [Bc × D_HEAD] FP16 = 8 KB
 *   weight_smem [Br_BLOCK × Bc] FP16 = 8 KB
 *
 * WMMA accumulator fragment layout (sm_86, m16n16k16):
 *   lane L holds 8 fp32 elements:
 *     groupID = L >> 2  (0..7) → row_lo = groupID, row_hi = groupID + 8
 *     in_group = L & 3  (0..3) → col_offset = in_group * 2
 *     x[0,1] → (row_lo, cols col_offset, col_offset+1)
 *     x[2,3] → (row_hi, cols col_offset, col_offset+1)
 *     x[4,5] → (row_lo, cols col_offset+8, col_offset+9)
 *     x[6,7] → (row_hi, cols col_offset+8, col_offset+9)
 *
 * Per-row max reduction:
 *   row_lo cols 0..15 of one tile are split: 4 lanes (in_group 0..3) × 4 cols each
 *   (cols (0,1,8,9), (2,3,10,11), (4,5,12,13), (6,7,14,15))
 *   intra-group reduction via __shfl_xor_sync(_, _, 1) and (_, _, 2) covers all 4 lanes
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_regpv_lean_qcache_nosmem.sm_86.cubin \
 *        flash_attn_br16_v2.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16
#define Br_BLOCK    (NUM_WARPS * Br_WARP)
#define Bc          64
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)
#define TILES_Bc    (Bc / WMMA_N)

#define LOG2E   1.4426950408889634f
#define NEG_INF (-3.402823466e+38f)

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 3)
void flash_attn_br16_v2(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    extern __shared__ char smem_raw[];

    __half *K_tile      = (__half*)(smem_raw);
    __half *V_tile      = (__half*)(smem_raw + Bc * D_HEAD * sizeof(__half));
    __half *weight_smem = (__half*)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    int block_q_base  = blockIdx.x * Br_BLOCK;
    int warp_q_base   = block_q_base + warp_id * Br_WARP;
    int head_idx      = blockIdx.y;
    int batch_idx     = blockIdx.z;

    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;
    size_t base_offset  = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;

    const __half *Q_head = Q + base_offset;
    const __half *K_head = K + base_offset;
    const __half *V_head = V + base_offset;
    float        *O_head = O + base_offset;

    bool valid_warp = (warp_q_base < seq_len);

    __half *warp_weight = weight_smem + warp_id * Br_WARP * Bc;

    int groupID  = lane >> 2;       // 0..7
    int in_group = lane & 3;        // 0..3
    int row_lo   = groupID;         // 0..7
    int row_hi   = groupID + 8;     // 8..15
    int col_lo   = in_group * 2;    // 0,2,4,6
    int col_hi   = col_lo + 8;      // 8,10,12,14

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum[TILES_D];
    #pragma unroll
    for (int n = 0; n < TILES_D; n++) wmma::fill_fragment(pv_accum[n], 0.0f);

    // Lean per-thread softmax state
    float my_max_lo = NEG_INF;
    float my_max_hi = NEG_INF;
    float my_sum_lo = 0.0f;
    float my_sum_hi = 0.0f;

    // Q register cache
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag[TILES_D];
    if (valid_warp) {
        #pragma unroll
        for (int dk = 0; dk < TILES_D; dk++) {
            wmma::load_matrix_sync(q_frag[dk],
                Q_head + (size_t)warp_q_base * D_HEAD + dk * WMMA_K,
                D_HEAD);
        }
    }

    __syncthreads();

    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // Phase A: load K, V tiles
        for (int idx = global_thread; idx < Bc * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
            int kv_row    = idx / D_HEAD;
            int d_col     = idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            K_tile[idx] = (kv_global < seq_len) ? K_head[(size_t)kv_global * D_HEAD + d_col]
                                                  : __float2half(0.0f);
            V_tile[idx] = (kv_global < seq_len) ? V_head[(size_t)kv_global * D_HEAD + d_col]
                                                  : __float2half(0.0f);
        }
        __syncthreads();

        // Phase B: QK^T into score_frag (registers, no smem store)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag[TILES_Bc];

        if (valid_warp) {
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) wmma::fill_fragment(score_frag[n], 0.0f);

            #pragma unroll
            for (int dk = 0; dk < TILES_D; dk++) {
                #pragma unroll
                for (int n = 0; n < TILES_Bc; n++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(k_frag,
                        K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag[dk], k_frag, score_frag[n]);
                }
            }

            // Scale all score elements
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                #pragma unroll
                for (int e = 0; e < score_frag[n].num_elements; e++) {
                    score_frag[n].x[e] *= scale;
                }
            }
        }

        // ==============================================================
        // Phase C: on-fragment online softmax
        //
        // For each owned row (row_lo, row_hi), reduce max/sum across
        // the 4 lanes within the same row group via shfl_xor.
        // ==============================================================
        float rescale_lo = 1.0f, rescale_hi = 1.0f;

        if (valid_warp) {
            // Step 1: per-lane partial max for owned rows (across all TILES_Bc)
            float partial_max_lo = NEG_INF;
            float partial_max_hi = NEG_INF;

            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                partial_max_lo = fmaxf(partial_max_lo, score_frag[n].x[0]);
                partial_max_lo = fmaxf(partial_max_lo, score_frag[n].x[1]);
                partial_max_lo = fmaxf(partial_max_lo, score_frag[n].x[4]);
                partial_max_lo = fmaxf(partial_max_lo, score_frag[n].x[5]);
                partial_max_hi = fmaxf(partial_max_hi, score_frag[n].x[2]);
                partial_max_hi = fmaxf(partial_max_hi, score_frag[n].x[3]);
                partial_max_hi = fmaxf(partial_max_hi, score_frag[n].x[6]);
                partial_max_hi = fmaxf(partial_max_hi, score_frag[n].x[7]);
            }

            // Step 2: intra-group reduction (4 lanes per row group)
            partial_max_lo = fmaxf(partial_max_lo, __shfl_xor_sync(0xFFFFFFFF, partial_max_lo, 1));
            partial_max_lo = fmaxf(partial_max_lo, __shfl_xor_sync(0xFFFFFFFF, partial_max_lo, 2));
            partial_max_hi = fmaxf(partial_max_hi, __shfl_xor_sync(0xFFFFFFFF, partial_max_hi, 1));
            partial_max_hi = fmaxf(partial_max_hi, __shfl_xor_sync(0xFFFFFFFF, partial_max_hi, 2));
            // Now all 4 lanes in a group hold the same partial_max for their owned row.

            // Step 3: combine with running max, compute rescale factor
            float new_max_lo = fmaxf(my_max_lo, partial_max_lo);
            float new_max_hi = fmaxf(my_max_hi, partial_max_hi);
            rescale_lo = exp2f((my_max_lo - new_max_lo) * LOG2E);
            rescale_hi = exp2f((my_max_hi - new_max_hi) * LOG2E);
            my_max_lo = new_max_lo;
            my_max_hi = new_max_hi;

            // Step 4: compute exp weights, accumulate per-lane partial sums,
            //         write FP16 weights to smem at WMMA-row-major positions.
            float partial_sum_lo = 0.0f;
            float partial_sum_hi = 0.0f;

            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                int tile_col_base = n * WMMA_N;

                float w_lo0 = exp2f((score_frag[n].x[0] - new_max_lo) * LOG2E);
                float w_lo1 = exp2f((score_frag[n].x[1] - new_max_lo) * LOG2E);
                float w_lo4 = exp2f((score_frag[n].x[4] - new_max_lo) * LOG2E);
                float w_lo5 = exp2f((score_frag[n].x[5] - new_max_lo) * LOG2E);
                float w_hi2 = exp2f((score_frag[n].x[2] - new_max_hi) * LOG2E);
                float w_hi3 = exp2f((score_frag[n].x[3] - new_max_hi) * LOG2E);
                float w_hi6 = exp2f((score_frag[n].x[6] - new_max_hi) * LOG2E);
                float w_hi7 = exp2f((score_frag[n].x[7] - new_max_hi) * LOG2E);

                partial_sum_lo += w_lo0 + w_lo1 + w_lo4 + w_lo5;
                partial_sum_hi += w_hi2 + w_hi3 + w_hi6 + w_hi7;

                // Write FP16 weights at row-major (row, col) positions.
                __half *row_lo_ptr = warp_weight + row_lo * Bc + tile_col_base;
                __half *row_hi_ptr = warp_weight + row_hi * Bc + tile_col_base;
                row_lo_ptr[col_lo]     = __float2half(w_lo0);
                row_lo_ptr[col_lo + 1] = __float2half(w_lo1);
                row_lo_ptr[col_hi]     = __float2half(w_lo4);
                row_lo_ptr[col_hi + 1] = __float2half(w_lo5);
                row_hi_ptr[col_lo]     = __float2half(w_hi2);
                row_hi_ptr[col_lo + 1] = __float2half(w_hi3);
                row_hi_ptr[col_hi]     = __float2half(w_hi6);
                row_hi_ptr[col_hi + 1] = __float2half(w_hi7);
            }

            // Step 5: intra-group sum reduction
            partial_sum_lo += __shfl_xor_sync(0xFFFFFFFF, partial_sum_lo, 1);
            partial_sum_lo += __shfl_xor_sync(0xFFFFFFFF, partial_sum_lo, 2);
            partial_sum_hi += __shfl_xor_sync(0xFFFFFFFF, partial_sum_hi, 1);
            partial_sum_hi += __shfl_xor_sync(0xFFFFFFFF, partial_sum_hi, 2);

            // Step 6: update running_sum
            my_sum_lo = my_sum_lo * rescale_lo + partial_sum_lo;
            my_sum_hi = my_sum_hi * rescale_hi + partial_sum_hi;

            // Step 7: apply rescale to register-resident pv_accum
            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                pv_accum[n].x[0] *= rescale_lo;
                pv_accum[n].x[1] *= rescale_lo;
                pv_accum[n].x[2] *= rescale_hi;
                pv_accum[n].x[3] *= rescale_hi;
                pv_accum[n].x[4] *= rescale_lo;
                pv_accum[n].x[5] *= rescale_lo;
                pv_accum[n].x[6] *= rescale_hi;
                pv_accum[n].x[7] *= rescale_hi;
            }
        }

        __syncthreads();  // weight_smem ready for Phase D

        // Phase D: PV WMMA (unchanged)
        if (valid_warp) {
            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                #pragma unroll
                for (int k = 0; k < TILES_Bc; k++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                    wmma::load_matrix_sync(w_frag,
                        warp_weight + k * WMMA_K,
                        Bc);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        V_tile + k * WMMA_K * D_HEAD + n * WMMA_N,
                        D_HEAD);

                    wmma::mma_sync(pv_accum[n], w_frag, v_frag, pv_accum[n]);
                }
            }
        }

        __syncthreads();
    }

    // ================================================================
    // Final: write pv_accum * 1/running_sum directly to global O.
    // No smem staging.
    //
    // Each lane owns 8 elements per tile per 2 rows; lanes write directly
    // using the WMMA fragment layout mapping.
    // ================================================================
    if (valid_warp) {
        float rcp_lo = __frcp_rn(my_sum_lo);
        float rcp_hi = __frcp_rn(my_sum_hi);

        int g_row_lo = warp_q_base + row_lo;
        int g_row_hi = warp_q_base + row_hi;
        bool valid_lo = (g_row_lo < seq_len);
        bool valid_hi = (g_row_hi < seq_len);

        #pragma unroll
        for (int n = 0; n < TILES_D; n++) {
            int tile_col_base = n * WMMA_N;
            int gc_lo = tile_col_base + col_lo;
            int gc_hi = tile_col_base + col_hi;

            if (valid_lo) {
                O_head[(size_t)g_row_lo * D_HEAD + gc_lo]     = pv_accum[n].x[0] * rcp_lo;
                O_head[(size_t)g_row_lo * D_HEAD + gc_lo + 1] = pv_accum[n].x[1] * rcp_lo;
                O_head[(size_t)g_row_lo * D_HEAD + gc_hi]     = pv_accum[n].x[4] * rcp_lo;
                O_head[(size_t)g_row_lo * D_HEAD + gc_hi + 1] = pv_accum[n].x[5] * rcp_lo;
            }
            if (valid_hi) {
                O_head[(size_t)g_row_hi * D_HEAD + gc_lo]     = pv_accum[n].x[2] * rcp_hi;
                O_head[(size_t)g_row_hi * D_HEAD + gc_lo + 1] = pv_accum[n].x[3] * rcp_hi;
                O_head[(size_t)g_row_hi * D_HEAD + gc_hi]     = pv_accum[n].x[6] * rcp_hi;
                O_head[(size_t)g_row_hi * D_HEAD + gc_hi + 1] = pv_accum[n].x[7] * rcp_hi;
            }
        }
    }
}

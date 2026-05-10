/*
 * flash_attn_br16_regpv_pad.cu — Flash Attention Br=16 WMMA, register-PV, smem padding
 *
 * Variant of flash_attn_br16_regpv.cu with bank-conflict-free smem layouts
 * for ldmatrix.x4 access. Pad amounts configurable via -D flags:
 *
 *   -DKV_PAD_HALFS=8   pad K/V tile rows: 64→72 halfs (stride 144 B, mod 32 = 16 ✓)
 *                      WMMA-required: must be 0 or multiple of 8 halfs (16-B align)
 *
 *   -DSCORE_PAD_FLOATS=1   pad smem_work rows: 64→65 floats (stride 260 B, mod 32 = 4 ✓)
 *                          FP16 overlay view: 130 halfs stride. NOT WMMA-aligned —
 *                          we use load_matrix_sync only on the FP16 view, leading
 *                          dim must be multiple of 8 halfs → SCORE_PAD_FLOATS must
 *                          be multiple of 4 (i.e. 0 or 4 floats). For pad=1, the
 *                          weight load uses raw ldmatrix PTX which only requires
 *                          16-B alignment of the BASE pointer, not the leading dim.
 *                          → use SCORE_PAD_FLOATS ∈ {0, 4} for safety.
 *
 * Design summary (defaults: KV_PAD_HALFS=8, SCORE_PAD_FLOATS=4):
 *
 *   smem layout:
 *     K_tile [Bc=64 × STRIDE_K=72]   FP16 = 9216 B
 *     V_tile [Bc=64 × STRIDE_K=72]   FP16 = 9216 B
 *     smem_work [Br_BLOCK=64 × STRIDE_W=68]  FP32 = 17408 B
 *     total: 35840 B (35.0 KB) — 2 blocks/SM (vs 3 unpadded)
 *
 *   Bank conflict status (from scripts/audit/ldmatrix_conflicts.R):
 *     K/V (stride 144 B): gcd(36,32)=4 → 8/8 distinct banks ✓
 *     W   (stride 272 B): gcd(68,32)=4 → 8/8 distinct banks ✓ (FP16 overlay 272/2=136 halfs)
 *
 * Tradeoff: 1 block/SM occupancy loss vs 8× ldmatrix replay elimination.
 *
 * EMPIRICAL RESULT (2026-05-07, bench_br16_regpv_pad on RTX 3070 Ti):
 *   PADDING LOSES 20-32% across all sizes (seq 512..4096, batch*heads=128).
 *   Reason: 12-warp regime (3 blocks/SM) already hides 8× ldmatrix replays
 *   via warp scheduling. Dropping to 8 warps exposes per-warp serialization
 *   more than the conflict elimination saves.
 *
 *   Best of padded variants: kv8+w4 at seq=512 (essentially tied, 1.00×).
 *   Worst: w4-only at seq=2048 (0.68×).
 *
 *   Kept as counter-example. The path forward is XOR swizzle (no smem cost)
 *   or structural register-pressure reduction to enable padding without
 *   occupancy regression. See R helper scripts/audit/ldmatrix_conflicts.R
 *   (pad_tradeoff function) for the calibrated tradeoff model.
 *
 * Build (3 variants):
 *   nvcc --cubin -arch=sm_86 -O2 -DKV_PAD_HALFS=8  -DSCORE_PAD_FLOATS=4 \
 *        -o flash_br16_regpv_pad_kv8_w4.sm_86.cubin flash_attn_br16_regpv_pad.cu
 *   nvcc --cubin -arch=sm_86 -O2 -DKV_PAD_HALFS=8  -DSCORE_PAD_FLOATS=0 \
 *        -o flash_br16_regpv_pad_kv8_w0.sm_86.cubin flash_attn_br16_regpv_pad.cu
 *   nvcc --cubin -arch=sm_86 -O2 -DKV_PAD_HALFS=0  -DSCORE_PAD_FLOATS=4 \
 *        -o flash_br16_regpv_pad_kv0_w4.sm_86.cubin flash_attn_br16_regpv_pad.cu
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

// ---- Padding configuration (compile-time) ----
#ifndef KV_PAD_HALFS
#define KV_PAD_HALFS 8        // pad K/V tile row stride (in halfs); must be 0 or multiple of 8
#endif
#ifndef SCORE_PAD_FLOATS
#define SCORE_PAD_FLOATS 4    // pad smem_work row stride (in floats); 0 or multiple of 4 for WMMA
#endif

#define STRIDE_K (D_HEAD + KV_PAD_HALFS)         // halfs per K/V row
#define STRIDE_W (Bc + SCORE_PAD_FLOATS)         // floats per smem_work row
#define STRIDE_W_HALF (STRIDE_W * 2)             // halfs per smem_work row (FP16 overlay)

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 2)
void flash_attn_br16_regpv_pad(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + Bc * STRIDE_K * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * STRIDE_K * sizeof(__half));

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

    float *warp_work = smem_work + warp_id * Br_WARP * STRIDE_W;

    int groupID = lane >> 2;
    int row_lo  = groupID;
    int row_hi  = groupID + 8;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum[TILES_D];
    #pragma unroll
    for (int n = 0; n < TILES_D; n++) wmma::fill_fragment(pv_accum[n], 0.0f);

    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    __syncthreads();

    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // Phase A: load K_tile, V_tile (padded smem)
        for (int idx = global_thread; idx < Bc * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
            int kv_row    = idx / D_HEAD;
            int d_col     = idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            __half k_val = (kv_global < seq_len) ? K_head[(size_t)kv_global * D_HEAD + d_col]
                                                  : __float2half(0.0f);
            __half v_val = (kv_global < seq_len) ? V_head[(size_t)kv_global * D_HEAD + d_col]
                                                  : __float2half(0.0f);
            K_tile[kv_row * STRIDE_K + d_col] = k_val;
            V_tile[kv_row * STRIDE_K + d_col] = v_val;
        }
        __syncthreads();

        // Phase B: QK^T via WMMA
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
                        K_tile + n * WMMA_N * STRIDE_K + dk * WMMA_K,
                        STRIDE_K);
                    wmma::mma_sync(score_frag[n], q_frag, k_frag, score_frag[n]);
                }
            }

            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                #pragma unroll
                for (int elem_idx = 0; elem_idx < score_frag[n].num_elements; elem_idx++) {
                    score_frag[n].x[elem_idx] *= scale;
                }
                wmma::store_matrix_sync(
                    warp_work + n * WMMA_N,
                    score_frag[n],
                    STRIDE_W,
                    wmma::mem_row_major);
            }
        }

        __syncthreads();

        // Phase C: online softmax
        if (valid_warp) {
            float rescale_for_row[Br_WARP];

            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * STRIDE_W;

                float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                rescale_for_row[row] = rescale_factor;
                running_sum[row]    *= rescale_factor;

                float w_lo = exp2f((score_row[lane]             - new_max) * LOG2E);
                float w_hi = exp2f((score_row[lane + WARP_SIZE] - new_max) * LOG2E);

                // FP16 overlay: each row of warp_work is STRIDE_W floats = STRIDE_W*2 halfs.
                // Weight row r at offset (warp_work as half) + r * STRIDE_W_HALF
                __half *weight_row = (__half*)warp_work + row * STRIDE_W_HALF;
                weight_row[lane]             = __float2half(w_lo);
                weight_row[lane + WARP_SIZE] = __float2half(w_hi);

                float partial_sum = w_lo + w_hi;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                running_sum[row] += partial_sum;
            }

            // Apply rescale to register-resident pv_accum
            {
                float s_lo = rescale_for_row[row_lo];
                float s_hi = rescale_for_row[row_hi];
                #pragma unroll
                for (int n = 0; n < TILES_D; n++) {
                    pv_accum[n].x[0] *= s_lo;
                    pv_accum[n].x[1] *= s_lo;
                    pv_accum[n].x[2] *= s_hi;
                    pv_accum[n].x[3] *= s_hi;
                    pv_accum[n].x[4] *= s_lo;
                    pv_accum[n].x[5] *= s_lo;
                    pv_accum[n].x[6] *= s_hi;
                    pv_accum[n].x[7] *= s_hi;
                }
            }

            // Phase D: PV WMMA
            __half *weight_ptr = (__half*)warp_work;

            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                #pragma unroll
                for (int k = 0; k < TILES_Bc; k++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                    wmma::load_matrix_sync(w_frag,
                        weight_ptr + k * WMMA_K,
                        STRIDE_W_HALF);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        V_tile + k * WMMA_K * STRIDE_K + n * WMMA_N,
                        STRIDE_K);

                    wmma::mma_sync(pv_accum[n], w_frag, v_frag, pv_accum[n]);
                }
            }
        }

        __syncthreads();
    }

    // Finalize: store pv_accum, normalize, write output
    if (valid_warp) {
        #pragma unroll
        for (int n = 0; n < TILES_D; n++) {
            wmma::store_matrix_sync(
                warp_work + n * WMMA_N,
                pv_accum[n],
                STRIDE_W,
                wmma::mem_row_major);
        }
    }

    __syncthreads();

    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *src = warp_work + row * STRIDE_W;

            O_head[(size_t)global_q * D_HEAD + lane]             = src[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = src[lane + WARP_SIZE] * rcp_sum;
        }
    }
}

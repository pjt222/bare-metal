/*
 * cross_attn.cu — Cross-Attention for Diffusion Models (Q from image, K/V from text)
 *
 * In Stable Diffusion's UNet, cross-attention conditions the spatial features on text
 * embeddings (from CLIP). The key difference from self-attention:
 *
 *   Self-attention:   Q, K, V all from the same sequence  (seq_Q = seq_KV)
 *   Cross-attention:  Q from image features, K/V from text (seq_Q ≠ seq_KV)
 *
 * Shapes in Stable Diffusion:
 *   Q:  [batch, num_heads, H*W, D_head]   — image spatial tokens (H*W = 256..4096)
 *   K:  [batch, num_heads, seq_text, D_head] — CLIP text tokens  (seq_text = 77)
 *   V:  [batch, num_heads, seq_text, D_head]
 *   O:  [batch, num_heads, H*W, D_head]   — output, same shape as Q
 *
 * Cross-attention is a strict generalization of self-attention. The Flash Attention
 * algorithm is identical — just the KV tile loop runs over seq_KV instead of seq_Q.
 * No causal masking is needed (every image token attends to all text tokens).
 *
 * Formula:
 *   A = softmax( (Q @ K^T) / sqrt(D_head) )   [seq_Q × seq_KV]
 *   O = A @ V                                   [seq_Q × D_head]
 *
 * Key SASS instructions (same as flash_attn_br16):
 *   HMMA.16816.F32  — Q·K^T and weighted V accumulation (Tensor Cores)
 *   SHFL.BFLY       — per-row max + sum reduction (online softmax)
 *   MUFU.EX2        — exp2f for attention weights + rescale factors
 *   MUFU.RCP        — __frcp_rn for final normalization
 *
 * Kernel design:
 *   Same as flash_attn_br16 — Br_BLOCK=64 Q-rows per block, 4 warps × 16 rows.
 *   KV tile loop iterates over seq_kv (text) instead of seq_q (image).
 *   Shared memory layout and WMMA calls identical.
 *
 * Grid:  (ceil(seq_q / Br_BLOCK), num_heads, batch_size)
 * Block: (128, 1, 1) = 4 warps
 *
 * Shared memory: 48 KB (same as flash_attn_br16)
 *   K_tile:    [Bc × D_HEAD] FP16 =  8 KB
 *   V_tile:    [Bc × D_HEAD] FP16 =  8 KB
 *   smem_work: [Br_BLOCK × Bc] FP32 = 16 KB  (scores, then FP16 weights overlay)
 *   smem_pv:   [Br_BLOCK × D_HEAD] FP32 = 16 KB
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o cross_attn.sm_86.cubin cross_attn.cu
 *   cuobjdump -sass cross_attn.sm_86.cubin | grep HMMA
 *   → HMMA.16816.F32 (QK^T and PV, 16 calls each per warp per KV tile)
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16                          // Q rows per warp
#define Br_BLOCK    (NUM_WARPS * Br_WARP)       // = 64 Q rows per block
#define Bc          64                          // KV tile size
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD  / WMMA_K)          // = 4
#define TILES_Bc    (Bc      / WMMA_N)          // = 4

#define LOG2E  1.4426950408889634f

// -----------------------------------------------------------------------
// Kernel: cross_attn_br16
//
// Flash cross-attention: Q from image sequence (seq_q), K/V from text (seq_kv).
// Br=16 WMMA tiles per warp — identical to flash_attn_br16 except:
//   - seq_q  (Q dimension) and seq_kv (KV dimension) are independent
//   - KV tile loop runs 0..seq_kv instead of 0..seq_q
//   - No causal masking (image tokens attend to all text tokens)
//
// Inputs:
//   Q:  [batch × num_heads × seq_q   × D_HEAD] FP16
//   K:  [batch × num_heads × seq_kv  × D_HEAD] FP16
//   V:  [batch × num_heads × seq_kv  × D_HEAD] FP16
//   O:  [batch × num_heads × seq_q   × D_HEAD] FP32 output
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void cross_attn_br16(
    const __half * __restrict__ Q,      // [batch × heads × seq_q  × D_HEAD] FP16
    const __half * __restrict__ K,      // [batch × heads × seq_kv × D_HEAD] FP16
    const __half * __restrict__ V,      // [batch × heads × seq_kv × D_HEAD] FP16
    float        * __restrict__ O,      // [batch × heads × seq_q  × D_HEAD] FP32
    int   seq_q,                        // Q sequence length (image spatial: H*W)
    int   seq_kv,                       // KV sequence length (text: 77 for CLIP)
    int   num_heads,
    float scale                         // 1/sqrt(D_HEAD)
) {
    // ---- Shared memory (48 KB total) ----
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));
    float  *smem_pv   = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half)
                                           + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    // This block handles Q rows [block_q_base .. block_q_base + Br_BLOCK - 1]
    int block_q_base = blockIdx.x * Br_BLOCK;
    int warp_q_base  = block_q_base + warp_id * Br_WARP;
    int head_idx     = blockIdx.y;
    int batch_idx    = blockIdx.z;

    // Strides: Q uses seq_q, K/V use seq_kv
    size_t q_head_stride  = (size_t)seq_q  * D_HEAD;
    size_t kv_head_stride = (size_t)seq_kv * D_HEAD;
    size_t batch_q_stride = (size_t)num_heads * q_head_stride;
    size_t batch_kv_stride = (size_t)num_heads * kv_head_stride;

    const __half *Q_head = Q + batch_idx * batch_q_stride  + head_idx * q_head_stride;
    const __half *K_head = K + batch_idx * batch_kv_stride + head_idx * kv_head_stride;
    const __half *V_head = V + batch_idx * batch_kv_stride + head_idx * kv_head_stride;
    float        *O_head = O + batch_idx * batch_q_stride  + head_idx * q_head_stride;

    bool valid_warp = (warp_q_base < seq_q);

    // Per-warp smem section pointers
    float *warp_work = smem_work + warp_id * Br_WARP * Bc;
    float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;

    // Zero smem_pv (running output accumulator)
    for (int idx = global_thread; idx < Br_BLOCK * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
        smem_pv[idx] = 0.0f;
    }

    // Per-warp online softmax state (one value per Q row)
    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    __syncthreads();

    // ====================================================================
    // Main KV tile loop — iterates over text tokens (seq_kv)
    // ====================================================================
    for (int kv_base = 0; kv_base < seq_kv; kv_base += Bc) {

        // ==============================================================
        // Phase A: Load K_tile + V_tile from text sequence
        // Handles partial last tile (seq_kv not divisible by Bc).
        // ==============================================================
        for (int idx = global_thread; idx < Bc * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
            int kv_row    = idx / D_HEAD;
            int d_col     = idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            // Zero-pad if beyond seq_kv (handles CLIP's seq_kv=77 which needs padding to 128)
            __half k_val = (kv_global < seq_kv) ? K_head[(size_t)kv_global * D_HEAD + d_col]
                                                 : __float2half(0.0f);
            __half v_val = (kv_global < seq_kv) ? V_head[(size_t)kv_global * D_HEAD + d_col]
                                                 : __float2half(0.0f);
            K_tile[idx] = k_val;
            V_tile[idx] = v_val;
        }
        __syncthreads();

        // ==============================================================
        // Phase B: QK^T via WMMA → scores stored to warp_work [Br_WARP × Bc]
        //
        // Q tile loaded from global memory (avoids storing Q in smem).
        // K tile loaded col_major → automatic K^T without explicit transpose.
        // ==============================================================
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

            // Scale and store scores to smem
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

        // ==============================================================
        // Phase C: Online softmax + PV accumulation
        //
        // KV padding mask: padded positions (kv_base + col >= seq_kv) must be
        // set to -infinity BEFORE the softmax max/exp computation.
        // Why: K_tile is zero-padded, so scores = Q @ 0^T = 0. But exp(0 - max)
        // is NOT zero — it inflates the softmax denominator, producing incorrect
        // output magnitudes. Setting score to -inf forces exp(-inf - max) = 0.
        //
        // SHFL.BFLY: 5 rounds per row for max and sum reductions.
        // ==============================================================
        if (valid_warp) {
            // Precompute per-lane mask for this KV tile
            // lane covers column index 'lane' and 'lane + WARP_SIZE' within [0, Bc)
            const float NEG_INF = -3.402823466e+38f;
            bool lo_padded = ((kv_base + (int)lane)             >= seq_kv);
            bool hi_padded = ((kv_base + (int)lane + WARP_SIZE) >= seq_kv);

            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;
                float *pv_row    = warp_pv   + row * D_HEAD;

                // Read scores into registers; apply -inf mask for padded KV positions
                float score_lo = lo_padded ? NEG_INF : score_row[lane];
                float score_hi = hi_padded ? NEG_INF : score_row[lane + WARP_SIZE];

                // Row max over (possibly masked) scores
                float partial_max = fmaxf(score_lo, score_hi);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                // Rescale running output by exp(old_max - new_max)
                pv_row[lane]             *= rescale_factor;
                pv_row[lane + WARP_SIZE] *= rescale_factor;

                // Attention weights: exp(score - max) using masked local scores
                // Padded positions: score_lo/hi = -inf → exp(-inf) = 0 → w = 0 ✓
                float w_lo = exp2f((score_lo - new_max) * LOG2E);
                float w_hi = exp2f((score_hi - new_max) * LOG2E);

                __half *weight_row = (__half*)warp_work + row * Bc;
                weight_row[lane]             = __float2half(w_lo);
                weight_row[lane + WARP_SIZE] = __float2half(w_hi);

                // Running sum update
                float partial_sum = w_lo + w_hi;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                running_sum[row] += partial_sum;
            }

            // ==============================================================
            // Phase D: PV WMMA — attention_weights @ V_tile → smem_pv update
            // ==============================================================
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

        __syncthreads();
    }

    // ====================================================================
    // Finalize: O[q_row] = smem_pv[row] / running_sum[row]
    // ====================================================================
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_q) break;

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *pv_row = warp_pv + row * D_HEAD;

            O_head[(size_t)global_q * D_HEAD + lane]             = pv_row[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = pv_row[lane + WARP_SIZE] * rcp_sum;
        }
    }
}

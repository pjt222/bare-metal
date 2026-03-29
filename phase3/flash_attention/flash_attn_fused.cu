/*
 * flash_attn_fused.cu — Flash Attention accepting [B,S,H,D] layout directly
 *
 * Accepts FP16 Q/K/V in [B, S, H, D] layout and writes FP32 output in
 * [B, S, H, D] layout. This eliminates all transpose kernels from the
 * attention pipeline — the HGEMM output ([BS×D] = [B,S,H,D]) can be
 * converted to FP16 in-place and fed directly to this kernel, and the
 * output can be converted to FP16 and fed directly to the output HGEMM.
 *
 * Eliminates 4 kernel launches from the attention pipeline:
 *   - 3× transpose_bshd [B,S,H,D]→[B,H,S,D] (for Q, K, V)
 *   - 1× transpose_bhsd [B,H,S,D]→[B,S,H,D] (for output O)
 *
 * Differences from flash_attn_br16:
 *   - Input:  FP16 [B,S,H,D] instead of FP16 [B,H,S,D]
 *   - Output: FP32 [B,S,H,D] instead of FP32 [B,H,S,D]
 *   - Row stride between sequence positions = d_model (not D_HEAD)
 *   - Q loaded from global with stride d_model (WMMA handles this)
 *   - K/V tile loading uses stride d_model for global address
 *   - Smem: 48 KB (unchanged from br16 — Q loaded from global)
 *   - K/V bandwidth: unchanged (FP16 loads, same as br16)
 *
 * Q loading: with BSHD layout, Q's row stride is d_model (vs D_HEAD in
 * BHSD). Each head's D_HEAD FP16 values are still contiguous (128 bytes
 * = one L2 cache line), but rows are d_model apart. The WMMA load_matrix_sync
 * handles this via its leading_dim parameter.
 *
 * NOTE: Adding Q_tile to smem (56 KB) was tested and caused 2× regression.
 * On GA104 sm_86, max smem per SM is 100 KB. At 56 KB/block, only 1 block
 * fits (2×56=112 > 100). The real cliff is 50 KB/block, not 64 KB.
 *
 * Shared memory layout (48 KB — same as br16):
 *   K_tile:    [Bc × D_HEAD]       FP16 = [64×64×2] =  8 KB
 *   V_tile:    [Bc × D_HEAD]       FP16 = [64×64×2] =  8 KB
 *   smem_work: [Br_BLOCK × Bc]     FP32 = [64×64×4] = 16 KB
 *   smem_pv:   [Br_BLOCK × D_HEAD] FP32 = [64×64×4] = 16 KB
 *
 * Grid:  (ceil(seq_len / Br_BLOCK), num_heads, batch_size)
 * Block: (128, 1, 1) = 4 warps
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_fused.sm_86.cubin flash_attn_fused.cu
 *   cuobjdump -sass flash_fused.sm_86.cubin | grep HMMA
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16
#define Br_BLOCK    (NUM_WARPS * Br_WARP)      // = 64
#define Bc          64
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)          // = 4
#define TILES_Bc    (Bc / WMMA_N)              // = 4

#define LOG2E   1.4426950408889634f

// -----------------------------------------------------------------------
// Kernel: flash_attn_fused
//
// Same algorithm as flash_attn_br16 — online softmax, WMMA QK^T and PV.
// Only the global memory addressing differs ([B,S,H,D] strides).
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_fused(
    const __half * __restrict__ Q,   // [batch × seq_len × num_heads × D_HEAD] FP16
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,   // [batch × seq_len × num_heads × D_HEAD] FP32
    int   seq_len,
    int   num_heads,
    float scale
) {
    // ---- Shared memory (48 KB — same as br16) ----
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + 1 * Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));
    float  *smem_pv   = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half)
                                           + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;               // 0..127
    int warp_id       = global_thread / WARP_SIZE; // 0..3
    int lane          = global_thread % WARP_SIZE; // 0..31

    int block_q_base  = blockIdx.x * Br_BLOCK;
    int warp_q_base   = block_q_base + warp_id * Br_WARP;
    int head_idx      = blockIdx.y;
    int batch_idx     = blockIdx.z;

    // [B, S, H, D] strides:
    //   element(b,s,h,d) = b * seq_len * d_model + s * d_model + h * D_HEAD + d
    //   Row stride between consecutive s: d_model
    //   Head offset: head_idx * D_HEAD
    int d_model = num_heads * D_HEAD;
    size_t batch_offset = (size_t)batch_idx * seq_len * d_model;
    size_t head_offset  = (size_t)head_idx * D_HEAD;

    // Q base pointer for WMMA loads: stride d_model between rows
    const __half *Q_bshd = Q + batch_offset + head_offset;

    bool valid_warp = (warp_q_base < seq_len);

    float *warp_work = smem_work + warp_id * Br_WARP * Bc;
    float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;

    // ---- Initialize smem_pv to 0 ----
    for (int idx = global_thread; idx < Br_BLOCK * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
        smem_pv[idx] = 0.0f;
    }

    // ---- Per-warp online softmax state ----
    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    __syncthreads();  // smem_pv zeroed

    // ====================================================================
    // Main KV tile loop
    // ====================================================================
    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // ==============================================================
        // Phase A: Load K_tile + V_tile from FP16 [B,S,H,D] global
        //   → contiguous FP16 smem [Bc × D_HEAD]
        //
        // Global addr: batch_offset + (kv_base + kv_row) * d_model + head_offset + d_col
        // Smem addr:   idx = kv_row * D_HEAD + d_col (contiguous)
        // ==============================================================
        for (int idx = global_thread; idx < Bc * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
            int kv_row    = idx / D_HEAD;
            int d_col     = idx % D_HEAD;
            int kv_global = kv_base + kv_row;
            if (kv_global < seq_len) {
                size_t addr = batch_offset + (size_t)kv_global * d_model + head_offset + d_col;
                K_tile[idx] = K[addr];
                V_tile[idx] = V[addr];
            } else {
                K_tile[idx] = __float2half(0.0f);
                V_tile[idx] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ==============================================================
        // Phase B: QK^T via WMMA → scores in warp_work (FP32)
        //
        // Q loaded from global [B,S,H,D] with stride d_model.
        // ==============================================================
        if (valid_warp) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag[TILES_Bc];
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) wmma::fill_fragment(score_frag[n], 0.0f);

            #pragma unroll
            for (int dk = 0; dk < TILES_D; dk++) {
                // Q_frag from global: [B,S,H,D], stride d_model between rows
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
                wmma::load_matrix_sync(q_frag,
                    Q_bshd + (size_t)warp_q_base * d_model + dk * WMMA_K,
                    d_model);

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
        // Phase C: Online softmax + Phase D: PV WMMA accumulation
        // (Identical to flash_attn_br16 — operates entirely in smem)
        // ==============================================================
        if (valid_warp) {

            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;
                float *pv_row    = warp_pv   + row * D_HEAD;

                float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                pv_row[lane]             *= rescale_factor;
                pv_row[lane + WARP_SIZE] *= rescale_factor;
                running_sum[row]         *= rescale_factor;

                float w_lo = exp2f((score_row[lane]             - new_max) * LOG2E);
                float w_hi = exp2f((score_row[lane + WARP_SIZE] - new_max) * LOG2E);

                __half *weight_row = (__half*)warp_work + row * Bc;
                weight_row[lane]             = __float2half(w_lo);
                weight_row[lane + WARP_SIZE] = __float2half(w_hi);

                float partial_sum = w_lo + w_hi;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                running_sum[row] += partial_sum;
            }

            // PV WMMA: weight_frag[16×Bc] × V_tile[Bc×D_HEAD] → smem_pv
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
    // Finalize: normalize smem_pv and store to FP32 [B,S,H,D] output
    // ====================================================================
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *pv_row = warp_pv + row * D_HEAD;

            // Write to [B, S, H, D]: base = batch*S*d_model + seq*d_model + head*D + d
            size_t out_base = batch_offset + (size_t)global_q * d_model + head_offset;
            O[out_base + lane]             = pv_row[lane]             * rcp_sum;
            O[out_base + lane + WARP_SIZE] = pv_row[lane + WARP_SIZE] * rcp_sum;
        }
    }
}

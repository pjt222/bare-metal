/*
 * flash_attn_br16_bc128.cu — Flash Attention Br=16 WMMA, Bc=128 (doubled KV tile)
 *
 * Upgrade from flash_attn_br16 (Bc=64): tile the KV dimension at 128 instead of 64.
 *
 * Why Bc=128:
 *   At Bc=64 with seq=1024:
 *     - 16 KV tile iterations per block
 *     - 32 HMMA per warp per tile (QK^T + PV: 4×4 + 4×4)
 *     - Tile-load overhead (~20% of cycles): __syncthreads + LDG × 4 rounds
 *   At Bc=128 with seq=1024:
 *     - 8 KV tile iterations per block (half as many)
 *     - 64 HMMA per warp per tile (QK^T: 4×8 + PV: 4×8)
 *     - Tile-load overhead halved in relative terms (same absolute cost, 2× longer compute)
 *
 * HMMA counts per warp per tile (Bc=128):
 *   QK^T: TILES_D × TILES_Bc = 4 × 8 = 32  (vs 4×4=16 at Bc=64)
 *   PV:   TILES_D × TILES_Bc = 4 × 8 = 32  (vs 4×4=16 at Bc=64)
 *   Total: 64 HMMA per warp per tile          (vs 32 at Bc=64)
 *
 * Online softmax adapts to Bc=128:
 *   Each lane now covers 4 positions per row: [lane], [lane+32], [lane+64], [lane+96]
 *   The SHFL.BFLY reduction is identical (still 5 rounds within the warp).
 *   Running sum update adds 4 exp() values instead of 2.
 *
 * Shared memory (80 KB — requires cuFuncSetAttribute):
 *   K_tile:    [Bc × D_HEAD] FP16 = [128×64×2] = 16 KB  (was 8 KB)
 *   V_tile:    [Bc × D_HEAD] FP16 = [128×64×2] = 16 KB  (was 8 KB)
 *   smem_work: [Br_BLOCK × Bc] FP32 = [64×128×4] = 32 KB (was 16 KB — scores + FP16 weights)
 *   smem_pv:   [Br_BLOCK × D_HEAD] FP32 = [64×64×4] = 16 KB  (unchanged)
 *   Total: 80 KB
 *
 * Note: double-buffering Bc=128 tiles would require 2×(16+16)+32+16 = 112 KB > 99 KB max.
 * So tiles are single-buffered. Warp interleaving provides the latency hiding.
 *
 * Constraint: seq_len must be a multiple of Bc=128.
 *   (seq_len % 128 == 0: valid for most transformer configs — 256, 512, 1024, 2048, etc.)
 *
 * SASS instructions:
 *   HMMA.16816.F32  — 64 per warp per tile (2× more than Bc=64)
 *   SHFL.BFLY       — 5 rounds per row (same as Bc=64)
 *   MUFU.EX2        — exp2f for weights and rescale (unchanged)
 *   MUFU.RCP        — final normalization (unchanged)
 *   LDG.E           — tile loads (same per tile; fewer tiles overall)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 \
 *        -o flash_br16_bc128.sm_86.cubin flash_attn_br16_bc128.cu
 *   cuobjdump -sass flash_br16_bc128.sm_86.cubin | grep HMMA | wc -l
 *   → expect ~128 HMMA (2× the 64 of Bc=64 kernel, due to doubled TILES_Bc)
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16
#define Br_BLOCK    (NUM_WARPS * Br_WARP)   // = 64

// ---- KEY CHANGE: Bc doubled from 64 to 128 ----
#define Bc          128

#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)       // = 4 (unchanged)
#define TILES_Bc    (Bc     / WMMA_N)       // = 8 (was 4)
#define LOG2E       1.4426950408889634f

// ---- Shared memory layout sizes ----
// K_tile and V_tile are now Bc×D_HEAD = 128×64 FP16 = 16 KB each
// smem_work per block: Br_BLOCK × Bc × 4 = 64 × 128 × 4 = 32 KB (was 16 KB)
// smem_pv:             Br_BLOCK × D_HEAD × 4 = 64 × 64 × 4 = 16 KB (unchanged)
// Total: 16 + 16 + 32 + 16 = 80 KB

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_bc128(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,        // must be divisible by Bc=128
    int   num_heads,
    float scale
) {
    extern __shared__ char smem_raw[];

    // ---- Shared memory layout (80 KB) ----
    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));
    float  *smem_pv   = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half)
                                           + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    int block_q_base = blockIdx.x * Br_BLOCK;
    int warp_q_base  = block_q_base + warp_id * Br_WARP;
    int head_idx     = blockIdx.y;
    int batch_idx    = blockIdx.z;

    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;
    size_t base_offset  = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;

    const __half *Q_head = Q + base_offset;
    const __half *K_head = K + base_offset;
    const __half *V_head = V + base_offset;
    float        *O_head = O + base_offset;

    bool valid_warp = (warp_q_base < seq_len);

    // Per-warp smem pointers
    float *warp_work = smem_work + warp_id * Br_WARP * Bc;
    float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;

    // Zero smem_pv
    for (int idx = global_thread; idx < Br_BLOCK * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
        smem_pv[idx] = 0.0f;
    }

    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    __syncthreads();

    // ================================================================
    // Main KV tile loop — Bc=128: 8 iterations at seq=1024 (vs 16)
    // ================================================================
    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // ==============================================================
        // Phase A: Load K_tile + V_tile [Bc × D_HEAD] = [128 × 64] FP16
        //
        // Total elements: 128 × 64 = 8192 FP16 per tile.
        // With 128 threads: 8192 / 128 = 64 elements per thread.
        // Loop stride = 128, runs 64 times.
        // ==============================================================
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

        // ==============================================================
        // Phase B: QK^T via WMMA — 32 HMMA per warp (vs 16 at Bc=64)
        //
        // score_frag[8]: covers [Br_WARP × Bc] = [16 × 128] scores per warp
        //   TILES_Bc = 8: 8 output column tiles of width WMMA_N=16
        //   TILES_D  = 4: 4 contraction tiles over D_HEAD=64
        //   Total: 8 × 4 = 32 HMMA calls (each HMMA.16816.F32)
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
                    // K loaded col_major: K_tile[n*16 : +16 rows, dk*16 : +16 cols]^T
                    // = K[kv+n*16+col][dk+k] → computes Q @ K^T
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(k_frag,
                        K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag, k_frag, score_frag[n]);
                }
            }

            // Scale and store to warp_work [16 × 128] FP32
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                #pragma unroll
                for (int elem_idx = 0; elem_idx < (int)score_frag[n].num_elements; elem_idx++) {
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
        // Phase C: Online softmax with Bc=128
        //
        // Each lane now covers 4 positions per row (stride 32):
        //   positions: lane, lane+32, lane+64, lane+96
        // Warp reduction (SHFL.BFLY) still uses 5 rounds (same as Bc=64).
        // FP16 weight overlay in first half of warp_work (8 KB).
        // ==============================================================
        if (valid_warp) {
            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row  = warp_work + row * Bc;
                float *pv_row     = warp_pv   + row * D_HEAD;

                // Read all 4 score values for this lane
                float score_0 = score_row[lane];
                float score_1 = score_row[lane + WARP_SIZE];
                float score_2 = score_row[lane + 2 * WARP_SIZE];
                float score_3 = score_row[lane + 3 * WARP_SIZE];

                // Max reduction across 128 scores via 32-thread warp SHFL
                float partial_max = fmaxf(fmaxf(score_0, score_1), fmaxf(score_2, score_3));
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max,
                                        __shfl_xor_sync(0xFFFFFFFF, partial_max, off));
                // partial_max is now the tile max for all 128 scores in this row

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                // Rescale smem_pv (D_HEAD=64 elements per row, 2 per lane)
                pv_row[lane]             *= rescale_factor;
                pv_row[lane + WARP_SIZE] *= rescale_factor;

                // Compute exp weights for all 4 score positions
                float w_0 = exp2f((score_0 - new_max) * LOG2E);
                float w_1 = exp2f((score_1 - new_max) * LOG2E);
                float w_2 = exp2f((score_2 - new_max) * LOG2E);
                float w_3 = exp2f((score_3 - new_max) * LOG2E);

                // Write FP16 weights overlaying FP32 scores in warp_work
                __half *weight_row = (__half*)warp_work + row * Bc;
                weight_row[lane]                 = __float2half(w_0);
                weight_row[lane + WARP_SIZE]     = __float2half(w_1);
                weight_row[lane + 2 * WARP_SIZE] = __float2half(w_2);
                weight_row[lane + 3 * WARP_SIZE] = __float2half(w_3);

                // Sum reduction across 128 weights
                float partial_sum = w_0 + w_1 + w_2 + w_3;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                running_sum[row] += partial_sum;
            }

            // ==============================================================
            // Phase D: PV via WMMA — 32 HMMA per warp (vs 16 at Bc=64)
            //
            // weight_frag [16 × 128] × V_tile [128 × 64] → [16 × 64] accumulated in smem_pv
            //   TILES_D   = 4: 4 output D_HEAD column tiles
            //   TILES_Bc = 8: 8 reduction tiles over Bc=128
            //   Total: 4 × 8 = 32 HMMA per warp per tile
            // ==============================================================
            __half *weight_ptr = (__half*)warp_work;

            #pragma unroll
            for (int n_d = 0; n_d < TILES_D; n_d++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum;
                wmma::load_matrix_sync(pv_accum,
                    warp_pv + n_d * WMMA_N,
                    D_HEAD,
                    wmma::mem_row_major);

                #pragma unroll
                for (int k_bc = 0; k_bc < TILES_Bc; k_bc++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                    wmma::load_matrix_sync(w_frag,
                        weight_ptr + k_bc * WMMA_K,
                        Bc);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        V_tile + k_bc * WMMA_K * D_HEAD + n_d * WMMA_N,
                        D_HEAD);

                    wmma::mma_sync(pv_accum, w_frag, v_frag, pv_accum);
                }

                wmma::store_matrix_sync(
                    warp_pv + n_d * WMMA_N,
                    pv_accum,
                    D_HEAD,
                    wmma::mem_row_major);
            }
        }

        __syncthreads();
    }

    // ---- Normalize and write output ----
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *pv_row = warp_pv + row * D_HEAD;

            O_head[(size_t)global_q * D_HEAD + lane]             = pv_row[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = pv_row[lane + WARP_SIZE] * rcp_sum;
        }
    }
}

/*
 * flash_attn_br16_regpv.cu — Flash Attention Br=16 WMMA with Register-Resident PV Accumulators
 *
 * Key optimization over flash_attn_br16.cu:
 *   Eliminates smem_pv (16 KB) by keeping PV accumulators as persistent WMMA
 *   fragments in registers across KV tiles. Per-row online softmax rescaling
 *   is applied directly to fragment elements using the sm_80+ WMMA accumulator
 *   layout (documented in PTX ISA for mma.sync.aligned.m16n8k16).
 *
 * Shared memory reduction: 48 KB → 32 KB
 *   On GA104 (100 KB smem/SM): floor(100/32) = 3 blocks/SM = 12 warps/SM
 *   vs baseline:               floor(100/48) = 2 blocks/SM =  8 warps/SM
 *   → +50% occupancy improvement
 *
 * smem traffic reduction per KV tile per warp:
 *   Baseline: 320 smem ops (64 rescale LDS/STS + 128 PV load + 128 PV store)
 *   This:      32 register FMULs (rescale in-place) + 0 PV smem load/store
 *   → ~10× less smem traffic for PV accumulation
 *
 * WMMA fragment layout (sm_86, verified empirically via verify_wmma_layout.cu):
 *   Each thread holds 8 floats covering 2 rows of the 16×16 output tile.
 *   groupID = lane >> 2 (0..7)
 *   row_lo = groupID        (rows 0..7)
 *   row_hi = groupID + 8    (rows 8..15)
 *   Elements x[0],x[1],x[4],x[5] → row_lo
 *   Elements x[2],x[3],x[6],x[7] → row_hi
 *   NOTE: This differs from the PTX mma.sync register layout! The WMMA API
 *   reorders elements internally. Verified on sm_86 with CUDA 12.8.
 *
 *   This layout is verified by the correctness test in bench_br16_regpv.cu.
 *   If NVIDIA changes the layout for a future architecture, the test will fail.
 *
 * Shared memory layout (32 KB):
 *   K_tile:    [Bc × D_HEAD]     FP16 = [64×64×2] =  8 KB
 *   V_tile:    [Bc × D_HEAD]     FP16 = [64×64×2] =  8 KB
 *   smem_work: [Br_BLOCK × Bc]   FP32 = [64×64×4] = 16 KB  (scores → FP16 weights)
 *   Total: 32 KB
 *
 * Register budget per thread (estimated):
 *   pv_accum[4]:    32 regs (persistent across KV tiles)
 *   running_max[16]: 16 regs (persistent)
 *   running_sum[16]: 16 regs (persistent)
 *   score_frag[4]:   32 regs (Phase B only, overlaps with Phase D temps)
 *   WMMA operands:   ~16 regs (q/k/w/v frags, loop-scoped)
 *   Misc temps:      ~14 regs
 *   Total peak:     ~126 regs (Phase B)
 *   At 3 blocks/SM: 384 threads × 126 regs = 48384 < 65536 ✓
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_regpv.sm_86.cubin flash_attn_br16_regpv.cu
 *   cuobjdump -sass flash_br16_regpv.sm_86.cubin | grep -E 'HMMA|SHFL|MUFU'
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
// Kernel: flash_attn_br16_regpv
//
// Functionally identical to flash_attn_br16. Structural change:
//   PV accumulators live in registers (WMMA fragments) instead of smem.
//   Per-row rescaling uses the sm_80+ fragment element-to-row mapping.
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 3)
void flash_attn_br16_regpv(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    // ---- Shared memory (32 KB total) ----
    // Layout:
    //   [0    ..  8KB)  K_tile:    [Bc × D_HEAD] FP16
    //   [8KB  .. 16KB)  V_tile:    [Bc × D_HEAD] FP16
    //   [16KB .. 32KB)  smem_work: [Br_BLOCK × Bc] FP32 (scores → FP16 weights)
    // No smem_pv — PV accumulators are register-resident.
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));

    int global_thread = threadIdx.x;               // 0..127
    int warp_id       = global_thread / WARP_SIZE;  // 0..3
    int lane          = global_thread % WARP_SIZE;  // 0..31

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

    // Per-warp section pointer into smem_work
    float *warp_work = smem_work + warp_id * Br_WARP * Bc;

    // ---- WMMA fragment layout: which 2 rows does this thread own? ----
    // Verified on sm_86: groupID maps to rows (groupID, groupID+8).
    // x[0,1,4,5] → row_lo, x[2,3,6,7] → row_hi.
    int groupID = lane >> 2;
    int row_lo  = groupID;       // rows 0..7
    int row_hi  = groupID + 8;   // rows 8..15

    // ---- Persistent register-resident PV accumulators ----
    // 4 WMMA fragments × 8 floats = 32 registers per thread.
    // These persist across all KV tile iterations.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum[TILES_D];
    #pragma unroll
    for (int n = 0; n < TILES_D; n++) wmma::fill_fragment(pv_accum[n], 0.0f);

    // ---- Per-warp running online softmax state ----
    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    // No smem_pv to zero — just barrier for any prior smem usage
    __syncthreads();

    // ====================================================================
    // Main KV tile loop
    // ====================================================================
    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // ==============================================================
        // Phase A: Load K_tile + V_tile (all 128 threads, coalesced)
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
        // Phase B: QK^T via WMMA → score_frag → warp_work (FP32)
        // Identical to flash_attn_br16.
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

            // Scale and store to warp_work
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

        __syncthreads();  // all warps' scores ready in smem_work

        // ==============================================================
        // Phase C: Online softmax + collect per-row rescale factors
        //
        // Same softmax as baseline. Key difference: instead of rescaling
        // smem_pv, we collect each row's rescale_factor and apply it to
        // the register-resident pv_accum after all rows are processed.
        // ==============================================================
        if (valid_warp) {

            // Collect per-row rescale factors. The 16-float array spills to
            // stack (64 bytes → L1), which is cheaper than 32 conditional moves
            // per KV tile in the hot softmax loop. Measured: +13% over cmov variant.
            float rescale_for_row[Br_WARP];

            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;

                // Max reduction across Bc=64 scores
                float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                // Store for deferred application after the row loop
                rescale_for_row[row] = rescale_factor;

                // Update running_sum
                running_sum[row] *= rescale_factor;

                // Compute exp weights and write as FP16 to warp_work (overlay)
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

            // ==============================================================
            // Apply rescale to register-resident pv_accum.
            // Verified layout (sm_86): x[0,1,4,5] → row_lo (groupID)
            //                          x[2,3,6,7] → row_hi (groupID + 8)
            // ==============================================================
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

            // ==============================================================
            // Phase D: PV WMMA — accumulate directly into register pv_accum.
            // No load_matrix_sync for C init. No store_matrix_sync for result.
            // pv_accum[n] persists in registers across KV tiles.
            // ==============================================================
            __half *weight_ptr = (__half*)warp_work;

            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
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

                    wmma::mma_sync(pv_accum[n], w_frag, v_frag, pv_accum[n]);
                }
            }
        }

        __syncthreads();  // all warps done — smem safe for next K/V tile
    }

    // ====================================================================
    // Finalize: store pv_accum to smem, normalize, write to global output.
    //
    // Reuse smem_work as output staging area. Since Bc == D_HEAD == 64,
    // the [Br_WARP × Bc] per-warp section has the same layout as
    // [Br_WARP × D_HEAD]. store_matrix_sync writes row-major with
    // leading dimension Bc = D_HEAD.
    // ====================================================================
    if (valid_warp) {
        // Store unnormalized pv_accum to smem_work
        #pragma unroll
        for (int n = 0; n < TILES_D; n++) {
            wmma::store_matrix_sync(
                warp_work + n * WMMA_N,
                pv_accum[n],
                Bc,                     // leading dimension = D_HEAD = 64
                wmma::mem_row_major);
        }
    }

    __syncthreads();  // ensure all stores visible before reads

    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *src = warp_work + row * Bc;

            O_head[(size_t)global_q * D_HEAD + lane]             = src[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = src[lane + WARP_SIZE] * rcp_sum;
        }
    }
}

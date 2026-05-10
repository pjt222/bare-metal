/*
 * flash_attn_br16.cu — Flash Attention with Br=16 per warp + HMMA.16816.F32
 *
 * This is the full WMMA Flash Attention kernel where Tensor Cores replace
 * both the Q·K score computation (QK^T) and the weighted V accumulation (PV).
 *
 * Key upgrade from flash_attn_4warp:
 *   flash_attn_4warp:  1 query row per warp, scalar FFMA for scores AND output.
 *   flash_attn_br16:   16 query rows per warp, HMMA for QK^T AND PV.
 *                      8× higher FLOP/instruction throughput for matmul steps.
 *
 * SASS instructions to observe:
 *   HMMA.16816.F32   — QK^T scores: Q[16×64] × K[64×64]^T → scores[16×64]
 *   HMMA.16816.F32   — PV output:   weights[16×64] × V[64×64] → delta[16×64]
 *   SHFL.BFLY        — per-row max and sum reductions for online softmax
 *   MUFU.EX2         — exp2f for attention weights + rescale factors
 *   MUFU.RCP         — final per-row normalization 1/running_sum
 *   LDS / STS        — shared memory access for K/V tiles + score/weight exchange
 *
 * Algorithm per block (64 query rows = 4 warps × 16 rows each):
 * ---------------------------------------------------------------
 *   1. Q_tile: load Q[block_q_base:+64, 0:64] into shared memory (once per block)
 *   2. For each KV tile [kv_base, kv_base+64):
 *        a. Load K_tile, V_tile into shared memory (all 128 threads, coalesced)
 *        b. Per warp: QK^T via WMMA → score_frag[4] = Q_warp[16×64] @ K_tile[64×64]^T
 *        c. Store score_frag to smem_scores[16×64] (warp's section)
 *        d. Per row: SHFL max reduction → new_max, rescale running_sum + smem_pv
 *        e. Compute exp weights in smem_scores (overlay as FP16)
 *        f. PV WMMA: load smem_pv as C, accumulate += weight_frag × V_frag
 *        g. Store updated C back to smem_pv
 *   3. Normalize smem_pv by running_sum, store to O
 *
 * Shared memory layout (48 KB):
 *   Q_tile:    [Br_BLOCK × D_HEAD] FP16 = [64×64×2] =  8 KB  (loaded once at start)
 *   K_tile:    [Bc × D_HEAD]       FP16 = [64×64×2] =  8 KB  (refreshed each KV tile)
 *   V_tile:    [Bc × D_HEAD]       FP16 = [64×64×2] =  8 KB  (refreshed each KV tile)
 *   smem_work: [Br_BLOCK × Bc]     FP32 = [64×64×4] = 16 KB  (scores → FP16 weights)
 *   smem_pv:   [Br_BLOCK × D_HEAD] FP32 = [64×64×4] = 16 KB  (running PV accumulator)
 *   Total: 56 KB → exceeds 48 KB limit.
 *
 *   Optimization: drop Q_tile from smem. Load Q directly from global into WMMA a_frag.
 *   Revised:
 *   K_tile:    8 KB
 *   V_tile:    8 KB
 *   smem_work: 16 KB  (FP32 scores, then reused as FP16 weights in first 8 KB)
 *   smem_pv:   16 KB
 *   Total: 48 KB ← exactly fits.
 *
 * WMMA tile dimensions: M=16, N=16, K=16 (maps to HMMA.16816.F32 in SASS)
 * Tiles for QK^T:  D_HEAD/K=4 reduction tiles × Bc/N=4 output column tiles = 16 HMMA calls
 * Tiles for PV:    Bc/K=4 reduction tiles × D_HEAD/N=4 output column tiles = 16 HMMA calls
 *
 * Inputs:
 *   Q, K, V: [batch × heads × seq_len × D_HEAD] FP16
 *   O:       [batch × heads × seq_len × D_HEAD] FP32
 *
 * Grid:  (ceil(seq_len / Br_BLOCK), num_heads, batch_size)
 * Block: (128, 1, 1) = 4 warps
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16.sm_86.cubin flash_attn_br16.cu
 *   cuobjdump -sass flash_br16.sm_86.cubin | grep HMMA
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
#define TILES_D     (D_HEAD / WMMA_K)          // = 4 (tiles over D_HEAD contraction)
#define TILES_Bc    (Bc / WMMA_N)              // = 4 (tiles over Bc output/contraction)

// log2(e) — use exp2f for MUFU.EX2 emission
#define LOG2E   1.4426950408889634f

// -----------------------------------------------------------------------
// Kernel: flash_attn_br16
//
// WMMA Flash Attention. Inputs are FP16; output is FP32.
// Each of the 4 warps independently handles 16 query rows using WMMA.
// Online softmax recurrence with per-row SHFL.BFLY reductions.
// -----------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_br16(
    const __half * __restrict__ Q,   // [batch × heads × seq_len × D_HEAD] FP16
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,   // [batch × heads × seq_len × D_HEAD] FP32
    int   seq_len,
    int   num_heads,
    float scale
) {
    // ---- Shared memory (48 KB total) ----
    // Layout (byte offsets):
    //   [0       ..  8KB)  K_tile:    [Bc × D_HEAD] FP16
    //   [8KB     .. 16KB)  V_tile:    [Bc × D_HEAD] FP16
    //   [16KB    .. 32KB)  smem_work: [Br_BLOCK × Bc] FP32 (scores → then FP16 weights)
    //   [32KB    .. 48KB)  smem_pv:   [Br_BLOCK × D_HEAD] FP32 (running output)
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + 1 * Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));
    float  *smem_pv   = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half)
                                           + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;               // 0..127
    int warp_id       = global_thread / WARP_SIZE; // 0..3
    int lane          = global_thread % WARP_SIZE; // 0..31

    // Block handles query rows [block_q_base .. block_q_base + Br_BLOCK - 1]
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

    // ---- Per-warp section pointers into smem ----
    float *warp_work = smem_work + warp_id * Br_WARP * Bc;      // [16 × Bc] FP32 (scores)
    float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;  // [16 × D_HEAD] FP32

    // ---- Initialize smem_pv to 0 (all 128 threads) ----
    for (int idx = global_thread; idx < Br_BLOCK * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
        smem_pv[idx] = 0.0f;
    }

    // ---- Initialize per-warp running online softmax state ----
    // Each array has Br_WARP=16 entries (one per query row of this warp).
    // All lanes in a warp hold identical values (redundant but correct).
    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    __syncthreads();  // ensure smem_pv is zeroed before KV loop

    // ====================================================================
    // Main KV tile loop
    // ====================================================================
    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // ==============================================================
        // Phase A: Load K_tile + V_tile (all 128 threads, coalesced)
        //   K_tile[Bc × D_HEAD]: thread t loads K[kv_base + t/D_HEAD][t%D_HEAD]
        //   Stride pattern: global_thread, +128, +256, ... covers Bc*D_HEAD=4096 FP16
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
        //
        // Computes: scores[Br_WARP × Bc] = Q[warp_q_base:+16, 0:64] @ K_tile[0:64, 0:64]^T
        //
        // WMMA tiling:
        //   Outer: n_tile ∈ [0..3]  → covers Bc=64 output columns (16 at a time)
        //   Inner: dk_tile ∈ [0..3] → covers D_HEAD=64 contraction (16 at a time)
        //   Total: 4 × 4 = 16 HMMA.16816.F32 calls per warp
        //
        // Each call: score_frag[n] += Q[q_rows, dk:+16] × K_tile[kv_n:+16, dk:+16]^T
        //   where col_major B gives the transpose: K_frag[k][n] = K_tile[kv_n+n][dk+k]
        // ==============================================================
        if (valid_warp) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag[TILES_Bc];
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) wmma::fill_fragment(score_frag[n], 0.0f);

            #pragma unroll
            for (int dk = 0; dk < TILES_D; dk++) {
                // A_frag: Q[warp_q_base:+16, dk*16:+16] — row_major, stride=D_HEAD
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
                wmma::load_matrix_sync(q_frag,
                    Q_head + (size_t)warp_q_base * D_HEAD + dk * WMMA_K,
                    D_HEAD);

                #pragma unroll
                for (int n = 0; n < TILES_Bc; n++) {
                    // B_frag: K_tile[n*16:+16, dk*16:+16] loaded col_major → K^T subtile
                    // col_major means: B_frag[k][col] = K_tile_ptr[col * leading_dim + k]
                    // = K_tile[(n*16 + col) * D_HEAD + dk*16 + k] = K[kv_base+n*16+col][dk*16+k]
                    // So WMMA computes C[m][col] += sum_k Q[row][dk+k] * K[kv+col][dk+k] ✓
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(k_frag,
                        K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag, k_frag, score_frag[n]);
                }
            }

            // Scale scores by 1/sqrt(D_HEAD) and store to warp_work [16 × Bc]
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                #pragma unroll
                for (int elem_idx = 0; elem_idx < score_frag[n].num_elements; elem_idx++) {
                    score_frag[n].x[elem_idx] *= scale;
                }
                wmma::store_matrix_sync(
                    warp_work + n * WMMA_N,   // col offset in [16 × Bc] row-major
                    score_frag[n],
                    Bc,                       // leading dimension
                    wmma::mem_row_major);
            }
        }

        __syncthreads();  // all warps' scores ready in smem_work

        // ==============================================================
        // Phase C: Online softmax — per row, then PV WMMA accumulation
        //
        // Thread lane covers columns [lane, lane+WARP_SIZE] for all 16 rows.
        // 32 threads × 2 = 64 = Bc. Per-row SHFL.BFLY max + sum reductions.
        // ==============================================================
        if (valid_warp) {

            // --- Per-row online softmax update ---
            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;
                float *pv_row    = warp_pv   + row * D_HEAD;

                // Max reduction across 64 scores (thread lane covers [lane, lane+32])
                float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));
                // partial_max is now the tile max for this row (same value in all 32 lanes)

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                // Rescale running output AND running sum by rescale_factor.
                // This corrects for the shift in the running maximum.
                // Online softmax recurrence: l_new = l_old * rescale + sum(exp(s - m_new))
                pv_row[lane]             *= rescale_factor;
                pv_row[lane + WARP_SIZE] *= rescale_factor;
                running_sum[row]         *= rescale_factor;

                // Compute exp weights and write back as FP16 into warp_work (overlay)
                float w_lo = exp2f((score_row[lane]             - new_max) * LOG2E);
                float w_hi = exp2f((score_row[lane + WARP_SIZE] - new_max) * LOG2E);

                // Write FP16 weights overlaying the FP32 score region.
                // warp_work occupies Br_WARP × Bc × 4 bytes = 16 × 64 × 4 = 4 KB per warp.
                // As FP16: Br_WARP × Bc × 2 = 2 KB — fits in the first half.
                __half *weight_row = (__half*)warp_work + row * Bc;
                weight_row[lane]             = __float2half(w_lo);
                weight_row[lane + WARP_SIZE] = __float2half(w_hi);

                // Running sum: sum of all Bc=64 exp weights for this row.
                // After SHFL.BFLY, all lanes hold the same wsum. Update is per-lane
                // but all lanes hold identical running_sum[row], so result is consistent.
                float partial_sum = w_lo + w_hi;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                running_sum[row] += partial_sum;
            }

            // --- Phase D: PV WMMA ---
            // weight_frag [16 × Bc] × V_tile [Bc × D_HEAD] → delta [16 × D_HEAD]
            // delta is accumulated INTO smem_pv (loaded as WMMA accumulator C init).
            //
            // WMMA tiling:
            //   Outer: n_tile ∈ [0..3] → output D_HEAD cols (16 at a time)
            //   Inner: k_tile ∈ [0..3] → contraction over Bc (16 at a time)
            //   Total: 4 × 4 = 16 HMMA.16816.F32 calls per warp

            __half *weight_ptr = (__half*)warp_work;  // FP16 weights in first 8 KB of warp_work

            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                // Load current smem_pv[warp_id][16 rows][n*16 : +16 cols] as accumulator C
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum;
                wmma::load_matrix_sync(
                    pv_accum,
                    warp_pv + n * WMMA_N,   // offset to col n*16 in [16 × D_HEAD] row-major
                    D_HEAD,
                    wmma::mem_row_major);

                #pragma unroll
                for (int k = 0; k < TILES_Bc; k++) {
                    // W_frag: weight_ptr[16 rows, k*16 : +16 cols] row_major, stride=Bc
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                    wmma::load_matrix_sync(w_frag,
                        weight_ptr + k * WMMA_K,
                        Bc);

                    // V_frag: V_tile[k*16 : +16 rows, n*16 : +16 cols] row_major, stride=D_HEAD
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        V_tile + k * WMMA_K * D_HEAD + n * WMMA_N,
                        D_HEAD);

                    // Tensor Core: pv_accum += w_frag × v_frag  → HMMA.16816.F32 in SASS
                    wmma::mma_sync(pv_accum, w_frag, v_frag, pv_accum);
                }

                // Store updated accumulator back to smem_pv
                wmma::store_matrix_sync(
                    warp_pv + n * WMMA_N,
                    pv_accum,
                    D_HEAD,
                    wmma::mem_row_major);
            }
        }

        __syncthreads();  // all warps done — smem safe to reuse for next K/V tile
    }

    // ====================================================================
    // Finalize: normalize smem_pv by running_sum and store to output
    //
    // Thread lane writes: O[warp_q_base + row][lane]        = pv_row[lane] / running_sum[row]
    //                     O[warp_q_base + row][lane+32]     = pv_row[lane+32] / sum
    // For all 16 rows of this warp.
    // ====================================================================
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;  // guard for non-multiple seq_len

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *pv_row = warp_pv + row * D_HEAD;

            O_head[(size_t)global_q * D_HEAD + lane]             = pv_row[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = pv_row[lane + WARP_SIZE]  * rcp_sum;
        }
    }
}

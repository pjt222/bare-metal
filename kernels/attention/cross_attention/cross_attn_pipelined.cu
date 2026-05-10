/*
 * cross_attn_pipelined.cu — Double-Buffered Cross-Attention with cp.async
 *
 * Problem with cross_attn_br16:
 *   K/V tiles are loaded synchronously with LDG instructions.
 *   The warp stalls ~300 cycles per tile waiting for DRAM.
 *   HMMA units sit idle during the entire load phase.
 *
 * Solution: cp.async (LDGSTS) + double-buffering
 *   While computing on K/V tile N (HMMA active), asynchronously prefetch
 *   tile N+1 into the alternate shared memory buffer.
 *   The DRAM latency (~300 cycles) is hidden inside the compute time (~200 cycles
 *   for 64 HMMA calls at Bc=64, seq_q=64 block).
 *
 * cp.async PTX instructions used:
 *   cp.async.cg.shared.global [smem], [gmem], 16
 *     Copies 16 bytes (8 FP16) from DRAM directly to smem, bypassing L1 cache.
 *     Asynchronous: warp does not stall, execution continues immediately.
 *   cp.async.commit_group
 *     Marks the end of the current group of pending async copies.
 *   cp.async.wait_group N
 *     Waits until all but N commit-groups have completed.
 *     wait_group 1 → current buffer ready, next buffer still in flight.
 *     wait_group 0 (= wait_all) → all buffers ready (used for last tile).
 *
 * The predicated form:
 *   cp.async.cg.shared.global [smem], [gmem], 16, src_size
 *     If src_size == 0: writes 16 zero bytes to smem (predicated out-of-bounds).
 *     Used to zero-pad the last partial KV tile (when seq_kv % Bc != 0).
 *
 * Shared memory layout (64 KB — requires cuFuncSetAttribute):
 *   K_tile[2]: [2 × Bc × D_HEAD] FP16 = 2 × 8 KB = 16 KB
 *   V_tile[2]: [2 × Bc × D_HEAD] FP16 = 2 × 8 KB = 16 KB
 *   smem_work: [Br_BLOCK × Bc]   FP32 = 64×64×4  = 16 KB  (scores → FP16 weights)
 *   smem_pv:   [Br_BLOCK × D_HEAD] FP32 = 64×64×4 = 16 KB
 *   Total: 64 KB
 *
 * SASS instructions expected:
 *   LDGSTS      — the cp.async instruction in SASS (async tile load)
 *   HMMA.16816.F32 — QK^T (16 calls/warp/tile) and PV (16 calls/warp/tile)
 *   SHFL.BFLY   — per-row max/sum for online softmax
 *   MUFU.EX2    — exp2f for attention weights + rescale
 *   MUFU.RCP    — final normalization
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o cross_attn_pipelined.sm_86.cubin cross_attn_pipelined.cu
 *   cuobjdump -sass cross_attn_pipelined.sm_86.cubin | grep -E 'LDGSTS|HMMA'
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// ---- Constants (same as cross_attn_br16) ----
#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16
#define Br_BLOCK    (NUM_WARPS * Br_WARP)  // = 64
#define Bc          64
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)      // = 4
#define TILES_Bc    (Bc     / WMMA_N)      // = 4
#define LOG2E       1.4426950408889634f

// ---- Shared memory sizes (in elements) ----
#define KV_BUF_ELEMS  (Bc * D_HEAD)        // = 4096 halfs = 8 KB per buffer
#define KV_BUF_BYTES  (KV_BUF_ELEMS * 2)  // = 8192 bytes


// ================================================================
// cp.async helper functions (PTX inline assembly)
// ================================================================

// Issue a 16-byte asynchronous copy from global to shared memory.
// Bypasses L1 (cache-global variant: .cg) — optimal for streaming tile loads.
// The src_size parameter enables predicated zero-fill:
//   src_size = 16 → copy 16 bytes from gmem_ptr
//   src_size = 0  → write 16 zero bytes to smem_ptr (out-of-bounds padding)
static __device__ __forceinline__
void cp_async16_pred(void* __restrict__ smem_ptr,
                     const void* __restrict__ gmem_ptr,
                     bool valid)
{
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    int src_sz = valid ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr),
           "l"((unsigned long long)gmem_ptr),
           "r"(src_sz)
        : "memory"
    );
}

// Mark the end of the current group of pending cp.async calls.
static __device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until all but 1 commit-group completes (double-buffer: current ready, next in flight).
static __device__ __forceinline__
void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

// Wait for ALL pending commit-groups (used at the last iteration).
static __device__ __forceinline__
void cp_async_wait0() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}


// ================================================================
// Tile prefetch helper: asynchronously load K_tile and V_tile for
// one KV iteration into the specified double-buffer slot (buf 0 or 1).
// Called by all 128 threads collaboratively.
// ================================================================
static __device__ __forceinline__
void prefetch_kv_tile(
    __half * __restrict__ smem_K,      // destination K buffer (K_tile[buf])
    __half * __restrict__ smem_V,      // destination V buffer (V_tile[buf])
    const __half * __restrict__ K_head, // K for this batch × head
    const __half * __restrict__ V_head, // V for this batch × head
    int kv_base,                        // starting KV row for this tile
    int seq_kv,                         // total KV sequence length
    int global_thread                   // threadIdx.x
) {
    // Each tile is Bc × D_HEAD FP16 = 4096 halfs = 8192 bytes = 512 × 16-byte chunks.
    // With 128 threads: 512/128 = 4 chunks per thread per tile.
    // Adjacent threads cover adjacent 16-byte chunks → coalesced DRAM access.

    for (int chunk = global_thread; chunk < KV_BUF_ELEMS / 8; chunk += NUM_WARPS * WARP_SIZE) {
        // chunk covers 8 FP16 elements = 16 bytes
        int elem_base = chunk * 8;              // first element index in tile
        int kv_row    = elem_base / D_HEAD;     // row within tile [0, Bc)
        int d_col     = elem_base % D_HEAD;     // column within D_HEAD

        int kv_global = kv_base + kv_row;
        bool row_valid = (kv_global < seq_kv);

        // Pointer arithmetic: point to valid row (row 0 if invalid — ptr ignored when src_sz=0)
        int safe_kv = row_valid ? kv_global : 0;
        const __half *K_src = &K_head[(size_t)safe_kv * D_HEAD + d_col];
        const __half *V_src = &V_head[(size_t)safe_kv * D_HEAD + d_col];

        cp_async16_pred(&smem_K[elem_base], K_src, row_valid);
        cp_async16_pred(&smem_V[elem_base], V_src, row_valid);
    }
}


// ================================================================
// Kernel: cross_attn_pipelined
//
// Flash cross-attention with cp.async double-buffering.
// Functionally identical to cross_attn_br16 but overlaps K/V tile
// loading with HMMA computation using asynchronous memory copies.
//
// Shared memory (64 KB):
//   K_tile[2]: double-buffered K tiles, [2][Bc × D_HEAD] FP16 = 16 KB
//   V_tile[2]: double-buffered V tiles, [2][Bc × D_HEAD] FP16 = 16 KB
//   smem_work: score matrix + FP16 weight overlay, [Br_BLOCK × Bc] FP32 = 16 KB
//   smem_pv:   running PV output accumulator,  [Br_BLOCK × D_HEAD] FP32 = 16 KB
//   Total: 64 KB
//
// Grid:  (ceil(seq_q / Br_BLOCK), num_heads, batch_size)
// Block: (128, 1, 1) = 4 warps
// ================================================================
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void cross_attn_pipelined(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_q,
    int   seq_kv,
    int   num_heads,
    float scale
) {
    // ---- Shared memory layout ----
    extern __shared__ char smem_raw[];

    // Double-buffered K and V tiles: [2][Bc × D_HEAD] halfs each
    __half *K_tile = (__half*)(smem_raw);
    __half *V_tile = (__half*)(smem_raw + 2 * KV_BUF_BYTES);

    // Score matrix and running PV (same layout as cross_attn_br16)
    float *smem_work = (float*)(smem_raw + 4 * KV_BUF_BYTES);
    float *smem_pv   = (float*)(smem_raw + 4 * KV_BUF_BYTES + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    int block_q_base = blockIdx.x * Br_BLOCK;
    int warp_q_base  = block_q_base + warp_id * Br_WARP;
    int head_idx     = blockIdx.y;
    int batch_idx    = blockIdx.z;

    size_t q_head_stride   = (size_t)seq_q  * D_HEAD;
    size_t kv_head_stride  = (size_t)seq_kv * D_HEAD;
    size_t batch_q_stride  = (size_t)num_heads * q_head_stride;
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

    // Per-warp online softmax state
    float running_max[Br_WARP];
    float running_sum[Br_WARP];
    #pragma unroll
    for (int row = 0; row < Br_WARP; row++) {
        running_max[row] = -3.402823466e+38f;
        running_sum[row] = 0.0f;
    }

    // ================================================================
    // PIPELINE PROLOGUE: prefetch tile 0 into buffer 0 before the main loop
    // ================================================================
    int num_kv_iters = (seq_kv + Bc - 1) / Bc;

    if (num_kv_iters > 0) {
        prefetch_kv_tile(
            K_tile + 0 * KV_BUF_ELEMS,   // K_tile[0]
            V_tile + 0 * KV_BUF_ELEMS,   // V_tile[0]
            K_head, V_head,
            0,                            // kv_base = 0 (first tile)
            seq_kv, global_thread
        );
        cp_async_commit();
    }

    __syncthreads();   // ensure smem_pv is zeroed before loop

    // ================================================================
    // MAIN KV TILE LOOP
    // ================================================================
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {

        int kv_base    = kv_iter * Bc;
        int cur_buf    = kv_iter & 1;          // alternates: 0, 1, 0, 1, ...
        int nxt_buf    = 1 - cur_buf;

        // ---- Prefetch NEXT tile into nxt_buf while we compute on cur_buf ----
        if (kv_iter + 1 < num_kv_iters) {
            prefetch_kv_tile(
                K_tile + nxt_buf * KV_BUF_ELEMS,
                V_tile + nxt_buf * KV_BUF_ELEMS,
                K_head, V_head,
                kv_base + Bc,                  // next tile's kv_base
                seq_kv, global_thread
            );
            cp_async_commit();
        }

        // ---- Wait for current buffer (cur_buf) to be ready ----
        // If a next tile is in flight (kv_iter+1 < num_kv_iters), wait_group 1 leaves
        // the next group pending. Otherwise wait_all to flush the last group.
        if (kv_iter + 1 < num_kv_iters) {
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();   // all threads see the freshly loaded K/V tile

        // Pointers to current buffer's K and V tiles
        const __half *cur_K_tile = K_tile + cur_buf * KV_BUF_ELEMS;
        const __half *cur_V_tile = V_tile + cur_buf * KV_BUF_ELEMS;

        // ==============================================================
        // Phase B: QK^T via WMMA → scores stored to warp_work [Br_WARP × Bc]
        // Identical to cross_attn_br16 Phase B, but reads from cur_K_tile.
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
                        cur_K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag, k_frag, score_frag[n]);
                }
            }

            // Scale and store scores to smem_work
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
        // Phase C: Online softmax with KV padding mask + PV accumulation
        // ==============================================================
        if (valid_warp) {
            const float NEG_INF = -3.402823466e+38f;

            // Precompute per-lane padding mask for this KV tile (O(1) per warp)
            bool lo_padded = ((kv_base + (int)lane)             >= seq_kv);
            bool hi_padded = ((kv_base + (int)lane + WARP_SIZE) >= seq_kv);

            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;
                float *pv_row    = warp_pv   + row * D_HEAD;

                // Read scores with -inf mask for padded positions
                float score_lo = lo_padded ? NEG_INF : score_row[lane];
                float score_hi = hi_padded ? NEG_INF : score_row[lane + WARP_SIZE];

                // Row max via SHFL.BFLY (5 rounds = log2(32))
                float partial_max = fmaxf(score_lo, score_hi);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max,
                                        __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                float new_max        = fmaxf(running_max[row], partial_max);
                float rescale_factor = exp2f((running_max[row] - new_max) * LOG2E);
                running_max[row]     = new_max;

                // Rescale running_sum and smem_pv
                running_sum[row] *= rescale_factor;
                for (int d = lane; d < D_HEAD; d += WARP_SIZE)
                    pv_row[d] *= rescale_factor;

                // Compute exp weights
                float w_lo = exp2f((score_lo - new_max) * LOG2E);
                float w_hi = exp2f((score_hi - new_max) * LOG2E);

                // Row sum via SHFL.BFLY
                float partial_sum = w_lo + w_hi;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);
                running_sum[row] += partial_sum;

                // Write FP16 weights into smem_work (overlay over FP32 scores)
                __half *weight_row = (__half*)warp_work + row * Bc;
                weight_row[lane]             = __float2half(w_lo);
                weight_row[lane + WARP_SIZE] = __float2half(w_hi);
            }
        }

        __syncthreads();

        // ==============================================================
        // Phase D: PV accumulation via WMMA
        // Load smem_pv as accumulator C, multiply weight_frag × V_frag, store back.
        // ==============================================================
        if (valid_warp) {
            #pragma unroll
            for (int n_d = 0; n_d < TILES_D; n_d++) {
                // Load current pv accumulator from smem
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum;
                wmma::load_matrix_sync(pv_accum,
                    warp_pv + n_d * WMMA_N,
                    D_HEAD,
                    wmma::mem_row_major);

                // Accumulate: pv_accum += weight_frag × V_frag (over Bc reduction dimension)
                #pragma unroll
                for (int bc_k = 0; bc_k < TILES_Bc; bc_k++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> weight_frag;
                    wmma::load_matrix_sync(weight_frag,
                        (__half*)warp_work + bc_k * WMMA_K,
                        Bc);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        cur_V_tile + bc_k * WMMA_K * D_HEAD + n_d * WMMA_N,
                        D_HEAD);

                    wmma::mma_sync(pv_accum, weight_frag, v_frag, pv_accum);
                }

                // Store updated accumulator back to smem_pv
                wmma::store_matrix_sync(
                    warp_pv + n_d * WMMA_N,
                    pv_accum,
                    D_HEAD,
                    wmma::mem_row_major);
            }
        }

        // Note: __syncthreads() is NOT needed here before the next iteration's
        // cp_async_wait, because the wait barrier + __syncthreads at the top of
        // the next iteration already provides the required ordering.
        // However, smem_work is written in Phase C and read in Phase D — both
        // within the same iteration and the same warp, so no inter-warp hazard.
        // We DO need a barrier before the next Phase B (which writes smem_work again).
        __syncthreads();
    }
    // ================================================================
    // End KV tile loop
    // ================================================================

    // ---- Normalize smem_pv by running_sum and write to global O ----
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            float inv_sum = __frcp_rn(running_sum[row]);  // MUFU.RCP
            float *pv_row = warp_pv + row * D_HEAD;
            float *o_row  = O_head  + (size_t)(warp_q_base + row) * D_HEAD;

            for (int d = lane; d < D_HEAD; d += WARP_SIZE) {
                o_row[d] = pv_row[d] * inv_sum;
            }
        }
    }
}

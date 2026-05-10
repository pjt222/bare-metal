/*
 * flash_attn_br16_pipeline.cu — Flash Attention Br=16 WMMA + cp.async Double-Buffering
 *
 * This is flash_attn_br16 with one targeted upgrade:
 *   K/V tile loads are replaced with cp.async (LDGSTS) instructions
 *   and the tile buffers are doubled (A/B ping-pong).
 *
 * Why this workload is the right target for cp.async (validated by postmortem):
 *   - self-attention seq_len=1024, Bc=64 → 16 KV tile iterations per block
 *   - K/V data is 1024 × 64 × 2 = 128 KB per head per batch — exceeds the 4 MB L2
 *     when multiple heads run concurrently (8 heads × 128 KB = 1 MB, fits in L2 but
 *     is warmed only for a few blocks before new head data evicts it)
 *   - The compute window per tile is ~200 cycles (64 HMMA calls)
 *   - DRAM latency is ~300 cycles → without overlap, 300 idle cycles per tile
 *   - With 16 tiles, total latency hidden: 16 × 300 = 4800 cycles = ~2.4 μs saved per block
 *   - contrast: cross-attn CLIP-77 had only 2 KV iterations — nothing to pipeline
 *
 * cp.async mechanism:
 *   cp.async.cg.shared.global [smem_dst], [gmem_src], 16
 *     Copies 16 bytes (8 FP16) from DRAM directly to smem, bypassing L1.
 *     The issuing warp does NOT stall — execution continues immediately.
 *     SASS: LDGSTS.E.BYPASS.128
 *
 *   cp.async.commit_group     marks end of async copy group
 *   cp.async.wait_group 1     wait for all but 1 group (leave next tile in flight)
 *   cp.async.wait_all         wait for all groups (last tile)
 *
 * Shared memory layout (64 KB — double-buffered K/V tiles):
 *   K_tile[2]: [2 × Bc × D_HEAD] FP16 = 2 × 8 KB = 16 KB
 *   V_tile[2]: [2 × Bc × D_HEAD] FP16 = 2 × 8 KB = 16 KB
 *   smem_work: [Br_BLOCK × Bc]   FP32 = 64 × 64 × 4 = 16 KB  (scores → FP16 weights)
 *   smem_pv:   [Br_BLOCK × D_HEAD] FP32 = 64 × 64 × 4 = 16 KB
 *   Total: 64 KB (requires cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES = 65536)
 *
 * Timeline diagram (per block, 16 KV tiles):
 *
 *   Prologue:
 *     [prefetch tile 0 → buf A]  (issued during smem_pv zeroing and init)
 *
 *   Iter 0 (buf=A):
 *     [prefetch tile 1 → buf B]
 *     wait_group 1               ← tile 0 in A is ready
 *     __syncthreads
 *     [Phase B: 64 HMMA on A]    ← tile 1 arriving in B concurrently
 *     [Phase C/D: softmax + PV]
 *     __syncthreads
 *
 *   Iter 1 (buf=B):
 *     [prefetch tile 2 → buf A]
 *     wait_group 1               ← tile 1 in B is ready
 *     __syncthreads
 *     [Phase B: 64 HMMA on B]    ← tile 2 arriving in A concurrently
 *     [Phase C/D: softmax + PV]
 *     ...
 *
 *   Iter 15 (last):
 *     (no prefetch)
 *     wait_all                   ← tile 15 is ready (last group)
 *     [Phase B: 64 HMMA]
 *     [Phase C/D: softmax + PV]
 *
 * SASS instructions (additional vs flash_attn_br16):
 *   LDGSTS.E.BYPASS.128  — async copy (replaces synchronous LDG for tile loads)
 *   ATOMS                — cp.async.commit_group / wait_group maps to barrier instructions
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 \
 *        -o flash_br16_pipeline.sm_86.cubin flash_attn_br16_pipeline.cu
 *   cuobjdump -sass flash_br16_pipeline.sm_86.cubin | grep -E 'LDGSTS|HMMA' | wc -l
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// ---- Constants (same as flash_attn_br16) ----
#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16
#define Br_BLOCK    (NUM_WARPS * Br_WARP)   // = 64
#define Bc          64
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)       // = 4
#define TILES_Bc    (Bc     / WMMA_N)       // = 4
#define LOG2E       1.4426950408889634f

// ---- Shared memory sizes ----
#define KV_BUF_ELEMS  (Bc * D_HEAD)         // = 4096 FP16 = 8 KB per buffer
#define KV_BUF_BYTES  (KV_BUF_ELEMS * 2)   // = 8192 bytes


// ================================================================
// cp.async PTX helpers — identical to cross_attn_pipelined.cu
// ================================================================

// 16-byte predicated async copy: if valid=false, writes 16 zeros to smem.
// The source pointer is still required (but ignored by hardware when src_sz=0).
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

static __device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

static __device__ __forceinline__
void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

static __device__ __forceinline__
void cp_async_wait0() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}


// ================================================================
// prefetch_kv_tile: async copy of one K/V tile into double-buffer slot.
// All 128 threads collaborate: KV_BUF_ELEMS/8 = 512 × 16-byte chunks total,
// so 512/128 = 4 cp_async calls per thread.
//
// Self-attention: Q, K, V all share the same seq_len.
// No partial tiles needed when seq_len is a multiple of Bc.
// The 'valid' flag handles the edge case if seq_len % Bc != 0.
// ================================================================
static __device__ __forceinline__
void prefetch_kv_tile(
    __half * __restrict__ smem_K,
    __half * __restrict__ smem_V,
    const __half * __restrict__ K_head,
    const __half * __restrict__ V_head,
    int kv_base,
    int seq_len,
    int global_thread
) {
    // KV_BUF_ELEMS / 8 = 512 sixteen-byte chunks per tile
    for (int chunk = global_thread; chunk < KV_BUF_ELEMS / 8; chunk += NUM_WARPS * WARP_SIZE) {
        int elem_base  = chunk * 8;
        int kv_row     = elem_base / D_HEAD;
        int d_col      = elem_base % D_HEAD;
        int kv_global  = kv_base + kv_row;
        bool row_valid = (kv_global < seq_len);

        // When invalid: src_sz=0 → hardware writes zeros, ignores the pointer
        int safe_kv = row_valid ? kv_global : 0;
        cp_async16_pred(&smem_K[elem_base], &K_head[(size_t)safe_kv * D_HEAD + d_col], row_valid);
        cp_async16_pred(&smem_V[elem_base], &V_head[(size_t)safe_kv * D_HEAD + d_col], row_valid);
    }
}


// ================================================================
// Kernel: flash_attn_br16_pipeline
//
// Functionally identical to flash_attn_br16.
// Difference: tile A/B double-buffer + cp.async overlaps DRAM→smem
// with the HMMA + softmax computation on the previous tile.
//
// Shared memory: 64 KB (requires cuFuncSetAttribute)
//   K_tile[2]: 16 KB   (double-buffered KV tiles)
//   V_tile[2]: 16 KB
//   smem_work: 16 KB   (scores FP32 → FP16 weight overlay)
//   smem_pv:   16 KB   (running PV accumulator)
//
// Grid:  (ceil(seq_len / Br_BLOCK), num_heads, batch_size)
// Block: (128, 1, 1) = 4 warps
// ================================================================
extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_br16_pipeline(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    // ---- Shared memory layout (64 KB) ----
    extern __shared__ char smem_raw[];

    // Double-buffered K and V tiles: [2][Bc × D_HEAD]
    __half *K_tile = (__half*)(smem_raw);
    __half *V_tile = (__half*)(smem_raw + 2 * KV_BUF_BYTES);

    // K_tile[buf] = K_tile + buf * KV_BUF_ELEMS
    // V_tile[buf] = V_tile + buf * KV_BUF_ELEMS

    // Score matrix and running PV accumulator (unchanged from baseline)
    float *smem_work = (float*)(smem_raw + 4 * KV_BUF_BYTES);
    float *smem_pv   = (float*)(smem_raw + 4 * KV_BUF_BYTES + Br_BLOCK * Bc * sizeof(float));

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

    float *warp_work = smem_work + warp_id * Br_WARP * Bc;
    float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;

    // Zero smem_pv (running PV accumulator)
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

    int num_kv_iters = (seq_len + Bc - 1) / Bc;

    // ================================================================
    // PIPELINE PROLOGUE: prefetch tile 0 into buffer 0
    // Issued here — overlaps with the smem_pv zeroing above.
    // ================================================================
    if (num_kv_iters > 0) {
        prefetch_kv_tile(
            K_tile + 0 * KV_BUF_ELEMS,   // K_tile[buf=0]
            V_tile + 0 * KV_BUF_ELEMS,   // V_tile[buf=0]
            K_head, V_head,
            0,                            // kv_base = 0 (first tile)
            seq_len, global_thread
        );
        cp_async_commit();
    }

    __syncthreads();   // smem_pv zeroed; prologue commit issued

    // ================================================================
    // MAIN KV TILE LOOP
    // ================================================================
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {

        int kv_base = kv_iter * Bc;
        int cur_buf = kv_iter & 1;     // alternates 0, 1, 0, 1, ...
        int nxt_buf = 1 - cur_buf;

        // ---- Prefetch NEXT tile into nxt_buf (overlaps with current compute) ----
        if (kv_iter + 1 < num_kv_iters) {
            prefetch_kv_tile(
                K_tile + nxt_buf * KV_BUF_ELEMS,
                V_tile + nxt_buf * KV_BUF_ELEMS,
                K_head, V_head,
                kv_base + Bc,           // next tile's kv_base
                seq_len, global_thread
            );
            cp_async_commit();
        }

        // ---- Wait for current buffer (cur_buf) to be ready ----
        // wait1: current tile ready, next tile still in flight → DRAM hidden by compute below
        // wait0: last iteration, flush all
        if (kv_iter + 1 < num_kv_iters) {
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();   // all threads see the freshly-arrived tile

        // Convenient pointers to current buffer's K and V
        const __half *cur_K = K_tile + cur_buf * KV_BUF_ELEMS;
        const __half *cur_V = V_tile + cur_buf * KV_BUF_ELEMS;

        // ==============================================================
        // Phase B: QK^T via WMMA (identical to baseline, using cur_K)
        // 16 HMMA.16816.F32 calls per warp.
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
                        cur_K + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag, k_frag, score_frag[n]);
                }
            }

            // Scale and store to smem_work
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
        // Phase C: Online softmax — per row, SHFL.BFLY max+sum reduction
        // ==============================================================
        if (valid_warp) {
            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;
                float *pv_row    = warp_pv   + row * D_HEAD;

                float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max,
                                        __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

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

            // ==============================================================
            // Phase D: PV accumulation via WMMA (using cur_V)
            // ==============================================================
            __half *weight_ptr = (__half*)warp_work;

            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum;
                wmma::load_matrix_sync(pv_accum,
                    warp_pv + n * WMMA_N,
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
                        cur_V + k_bc * WMMA_K * D_HEAD + n * WMMA_N,
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
    // ================================================================
    // End KV tile loop
    // ================================================================

    // ---- Normalize and write output ----
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;

            float rcp_sum = __frcp_rn(running_sum[row]);
            float *pv_row = warp_pv + row * D_HEAD;

            O_head[(size_t)global_q * D_HEAD + lane]             = pv_row[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = pv_row[lane + WARP_SIZE]  * rcp_sum;
        }
    }
}

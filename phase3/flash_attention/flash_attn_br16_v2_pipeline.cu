/*
 * flash_attn_br16_v2_pipeline.cu
 *   v2 (nosmem fragment-shfl) + cp.async double-buffered K/V prefetch.
 *
 * Builds on flash_attn_br16_v2.cu by adding cp.async overlap of K/V DRAM
 * loads with HMMA compute. The original flash_attn_br16_pipeline used a
 * 64 KB smem layout (double-buffered K/V + 16 KB smem_work + 16 KB smem_pv)
 * which forced 1 block/SM (4 warps). With v2's smem_work + smem_pv elimination,
 * the pipeline variant fits in 40 KB → 2 blocks/SM (8 warps).
 *
 * smem layout (40 KB):
 *   K_tile [2][Bc × D_HEAD] FP16 = 16 KB (double-buffered)
 *   V_tile [2][Bc × D_HEAD] FP16 = 16 KB (double-buffered)
 *   weight_smem [Br_BLOCK × Bc] FP16 = 8 KB
 *   Total: 40 KB → 2 blocks/SM (was 1 with original pipeline)
 *
 * Per the previous pipeline postmortem (gpu_reflections.md Insight 14):
 *   "cp.async benefit depends on compute/load ratio: helpful when short
 *    (8 IMMA/tile, +35%), harmful when long (64 HMMA/tile, -5%)."
 * The original pipeline lost 4-5% at 8 warps. With doubled occupancy (8
 * warps per SM × 2 blocks = 16 warps cumulative throughput potential), the
 * tradeoff may now favor cp.async. To be measured empirically.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2_pipeline.sm_86.cubin \
 *        flash_attn_br16_v2_pipeline.cu
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

#define KV_BUF_ELEMS  (Bc * D_HEAD)         // = 4096 FP16
#define KV_BUF_BYTES  (KV_BUF_ELEMS * 2)   // = 8192 bytes

// ================================================================
// cp.async PTX helpers
// ================================================================
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

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
static __device__ __forceinline__ void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}
static __device__ __forceinline__ void cp_async_wait0() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// All 128 threads collaborate: KV_BUF_ELEMS/8 = 512 chunks → 4 cp_async per thread.
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
    for (int chunk = global_thread; chunk < KV_BUF_ELEMS / 8; chunk += NUM_WARPS * WARP_SIZE) {
        int elem_base  = chunk * 8;
        int kv_row     = elem_base / D_HEAD;
        int d_col      = elem_base % D_HEAD;
        int kv_global  = kv_base + kv_row;
        bool row_valid = (kv_global < seq_len);
        int safe_kv = row_valid ? kv_global : 0;
        cp_async16_pred(&smem_K[elem_base], &K_head[(size_t)safe_kv * D_HEAD + d_col], row_valid);
        cp_async16_pred(&smem_V[elem_base], &V_head[(size_t)safe_kv * D_HEAD + d_col], row_valid);
    }
}

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 2)
void flash_attn_br16_v2_pipeline(
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
    __half *V_tile      = (__half*)(smem_raw + 2 * KV_BUF_BYTES);
    __half *weight_smem = (__half*)(smem_raw + 4 * KV_BUF_BYTES);

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

    int groupID  = lane >> 2;
    int in_group = lane & 3;
    int row_lo   = groupID;
    int row_hi   = groupID + 8;
    int col_lo   = in_group * 2;
    int col_hi   = col_lo + 8;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum[TILES_D];
    #pragma unroll
    for (int n = 0; n < TILES_D; n++) wmma::fill_fragment(pv_accum[n], 0.0f);

    float my_max_lo = NEG_INF, my_max_hi = NEG_INF;
    float my_sum_lo = 0.0f,    my_sum_hi = 0.0f;

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

    int num_kv_iters = (seq_len + Bc - 1) / Bc;

    // Pipeline prologue: prefetch tile 0 into buffer 0
    if (num_kv_iters > 0) {
        prefetch_kv_tile(
            K_tile + 0 * KV_BUF_ELEMS,
            V_tile + 0 * KV_BUF_ELEMS,
            K_head, V_head,
            0, seq_len, global_thread);
        cp_async_commit();
    }

    __syncthreads();

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {

        int kv_base = kv_iter * Bc;
        int cur_buf = kv_iter & 1;
        int nxt_buf = 1 - cur_buf;

        // Prefetch next tile (overlaps with current compute)
        if (kv_iter + 1 < num_kv_iters) {
            prefetch_kv_tile(
                K_tile + nxt_buf * KV_BUF_ELEMS,
                V_tile + nxt_buf * KV_BUF_ELEMS,
                K_head, V_head,
                kv_base + Bc, seq_len, global_thread);
            cp_async_commit();
        }

        // Wait for current buffer
        if (kv_iter + 1 < num_kv_iters) cp_async_wait1();
        else                            cp_async_wait0();
        __syncthreads();

        const __half *cur_K = K_tile + cur_buf * KV_BUF_ELEMS;
        const __half *cur_V = V_tile + cur_buf * KV_BUF_ELEMS;

        // Phase B: QK^T (registers, no smem store)
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
                        cur_K + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag[dk], k_frag, score_frag[n]);
                }
            }

            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) {
                #pragma unroll
                for (int e = 0; e < score_frag[n].num_elements; e++)
                    score_frag[n].x[e] *= scale;
            }
        }

        // Phase C: on-fragment softmax
        float rescale_lo = 1.0f, rescale_hi = 1.0f;

        if (valid_warp) {
            float partial_max_lo = NEG_INF, partial_max_hi = NEG_INF;

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

            partial_max_lo = fmaxf(partial_max_lo, __shfl_xor_sync(0xFFFFFFFF, partial_max_lo, 1));
            partial_max_lo = fmaxf(partial_max_lo, __shfl_xor_sync(0xFFFFFFFF, partial_max_lo, 2));
            partial_max_hi = fmaxf(partial_max_hi, __shfl_xor_sync(0xFFFFFFFF, partial_max_hi, 1));
            partial_max_hi = fmaxf(partial_max_hi, __shfl_xor_sync(0xFFFFFFFF, partial_max_hi, 2));

            float new_max_lo = fmaxf(my_max_lo, partial_max_lo);
            float new_max_hi = fmaxf(my_max_hi, partial_max_hi);
            rescale_lo = exp2f((my_max_lo - new_max_lo) * LOG2E);
            rescale_hi = exp2f((my_max_hi - new_max_hi) * LOG2E);
            my_max_lo = new_max_lo;
            my_max_hi = new_max_hi;

            float partial_sum_lo = 0.0f, partial_sum_hi = 0.0f;

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

            partial_sum_lo += __shfl_xor_sync(0xFFFFFFFF, partial_sum_lo, 1);
            partial_sum_lo += __shfl_xor_sync(0xFFFFFFFF, partial_sum_lo, 2);
            partial_sum_hi += __shfl_xor_sync(0xFFFFFFFF, partial_sum_hi, 1);
            partial_sum_hi += __shfl_xor_sync(0xFFFFFFFF, partial_sum_hi, 2);

            my_sum_lo = my_sum_lo * rescale_lo + partial_sum_lo;
            my_sum_hi = my_sum_hi * rescale_hi + partial_sum_hi;

            // Apply rescale to pv_accum
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

        __syncthreads();  // weight_smem ready

        // Phase D: PV WMMA
        if (valid_warp) {
            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                #pragma unroll
                for (int k = 0; k < TILES_Bc; k++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                    wmma::load_matrix_sync(w_frag,
                        warp_weight + k * WMMA_K, Bc);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        cur_V + k * WMMA_K * D_HEAD + n * WMMA_N, D_HEAD);

                    wmma::mma_sync(pv_accum[n], w_frag, v_frag, pv_accum[n]);
                }
            }
        }

        __syncthreads();
    }

    // Final: pv_accum / running_sum directly to global
    if (valid_warp) {
        float rcp_lo = __frcp_rn(my_sum_lo);
        float rcp_hi = __frcp_rn(my_sum_hi);

        int g_row_lo = warp_q_base + row_lo;
        int g_row_hi = warp_q_base + row_hi;
        bool valid_lo = (g_row_lo < seq_len);
        bool valid_hi = (g_row_hi < seq_len);

        #pragma unroll
        for (int n = 0; n < TILES_D; n++) {
            int gc_lo = n * WMMA_N + col_lo;
            int gc_hi = gc_lo + 8;

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

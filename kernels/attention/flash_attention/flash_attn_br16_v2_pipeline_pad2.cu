/*
 * flash_attn_br16_v2_pipeline_pad2.cu
 *   v2_pipeline + +8 padding on K_tile, V_tile, AND weight_smem.
 *   Extends pad1 (which only padded K/V) to also pad the weight tile.
 *
 *   weight_smem is [Br_BLOCK × Bc] = [64 × 64] FP16 = 8 KB. Row stride
 *   64 halfs = 128 bytes — same bank-period conflict as K/V. Padding
 *   stride to 72 → 144 B costs +1 KB. New total: 44 KB → 45 KB.
 */

/*
 * Original pad1 docstring follows for reference:
 *   v2_pipeline + +8 padding on K_tile/V_tile row stride.
 *
 * NCU profile of v2_pipeline (Observation U) showed:
 *   stall_mio       = 6.89 per-issue cycles
 *   stall_short_sb  = 5.41 per-issue cycles
 *   bank conflicts  = 88.1M / 100.7M load wavefronts = 87.5% conflict rate
 *
 * Root cause: K_tile and V_tile are [Bc × D_HEAD] = [64 × 64] FP16, row
 * stride = 128 bytes = exactly the smem bank period (32 banks × 4
 * bytes/bank). Every ldmatrix.x4 reads 8 rows landing on the same banks
 * → 8-way conflict.
 *
 * Fix: pad row stride to D_HEAD + 8 = 72 halfs = 144 bytes. gcd(144,128)
 * = 16, so 8 consecutive rows distribute across 16 distinct bank groups,
 * eliminating the conflict.
 *
 * Smem cost: K_tile per buffer 64×64×2 = 8 KB → 64×72×2 = 9 KB. Two
 * buffers each for K and V → +4 KB total. New total: 40 KB → 44 KB.
 * Still under 50 KB cliff for 2 blocks/SM.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_v2_pipeline_pad.sm_86.cubin \
 *        flash_attn_br16_v2_pipeline_pad.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define D_PAD       8                       // padding per row (NEW)
#define D_STRIDE    (D_HEAD + D_PAD)        // 72 halfs = 144 bytes (NEW)
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

// Padded buffer sizes (NEW)
#define KV_BUF_ELEMS  (Bc * D_STRIDE)         // = 64 × 72 = 4608 FP16 = 9216 B
#define KV_BUF_BYTES  (KV_BUF_ELEMS * 2)

// ----------------------------------------------------------------------
// cp.async helpers (unchanged from v2_pipeline)
// ----------------------------------------------------------------------
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

// ----------------------------------------------------------------------
// Padded prefetch: every cp.async loads 8 contiguous halfs (16 B) from
// global to a row-stride-72 smem layout. The 8 halfs per chunk all live
// in the same row, so global → padded smem is a non-strided copy, just
// to a destination address that jumps D_STRIDE per row instead of D_HEAD.
//
// Number of 8-half chunks per tile = Bc * D_HEAD / 8 = 64 * 64 / 8 = 512
// (exactly the same as the unpadded version — we don't store the pad
// bytes from global, we just leave them uninitialized).
// ----------------------------------------------------------------------
static __device__ __forceinline__
void prefetch_kv_tile_pad(
    __half * __restrict__ smem_K,
    __half * __restrict__ smem_V,
    const __half * __restrict__ K_head,
    const __half * __restrict__ V_head,
    int kv_base,
    int seq_len,
    int global_thread
) {
    const int chunks_per_tile = Bc * D_HEAD / 8;   // = 512
    for (int chunk = global_thread; chunk < chunks_per_tile; chunk += NUM_WARPS * WARP_SIZE) {
        int kv_row     = chunk / (D_HEAD / 8);     // 0..63
        int chunk_in_row = chunk % (D_HEAD / 8);   // 0..7
        int d_col      = chunk_in_row * 8;         // 0,8,16,...,56

        int kv_global  = kv_base + kv_row;
        bool row_valid = (kv_global < seq_len);
        int safe_kv = row_valid ? kv_global : 0;

        // Destination uses padded stride; source uses tight D_HEAD stride.
        int smem_off = kv_row * D_STRIDE + d_col;
        size_t gmem_off = (size_t)safe_kv * D_HEAD + d_col;

        cp_async16_pred(&smem_K[smem_off], &K_head[gmem_off], row_valid);
        cp_async16_pred(&smem_V[smem_off], &V_head[gmem_off], row_valid);
    }
}

#define W_PAD     8
#define W_STRIDE  (Bc + W_PAD)         // 72 halfs = 144 B (NEW for pad2)

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 2)
void flash_attn_br16_v2_pipeline_pad2(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    extern __shared__ char smem_raw[];

    __half *K_tile      = (__half*)(smem_raw);                              // [2][Bc × D_STRIDE]
    __half *V_tile      = (__half*)(smem_raw + 2 * KV_BUF_BYTES);          // [2][Bc × D_STRIDE]
    __half *weight_smem = (__half*)(smem_raw + 4 * KV_BUF_BYTES);          // [Br_BLOCK × W_STRIDE]

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

    __half *warp_weight = weight_smem + warp_id * Br_WARP * W_STRIDE;

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

    // Q register cache (Q is global, no padding change here)
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

    if (num_kv_iters > 0) {
        prefetch_kv_tile_pad(
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

        if (kv_iter + 1 < num_kv_iters) {
            prefetch_kv_tile_pad(
                K_tile + nxt_buf * KV_BUF_ELEMS,
                V_tile + nxt_buf * KV_BUF_ELEMS,
                K_head, V_head,
                kv_base + Bc, seq_len, global_thread);
            cp_async_commit();
        }

        if (kv_iter + 1 < num_kv_iters) cp_async_wait1();
        else                            cp_async_wait0();
        __syncthreads();

        const __half *cur_K = K_tile + cur_buf * KV_BUF_ELEMS;
        const __half *cur_V = V_tile + cur_buf * KV_BUF_ELEMS;

        // Phase B: QK^T — k_frag stride changes from D_HEAD to D_STRIDE
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag[TILES_Bc];

        if (valid_warp) {
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) wmma::fill_fragment(score_frag[n], 0.0f);

            #pragma unroll
            for (int dk = 0; dk < TILES_D; dk++) {
                #pragma unroll
                for (int n = 0; n < TILES_Bc; n++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                    // K is stored col-major-of-row-major via stride.
                    // n indexes which 16-row block; dk indexes which 16-col block.
                    // Row offset = n * WMMA_N * D_STRIDE; col offset = dk * WMMA_K
                    wmma::load_matrix_sync(k_frag,
                        cur_K + n * WMMA_N * D_STRIDE + dk * WMMA_K,
                        D_STRIDE);
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

        // Phase C: softmax (unchanged — operates on registers + weight_smem,
        // weight_smem is [Br_BLOCK × Bc] = [64 × 64] which has the same
        // bank-conflict issue but is much smaller and only written once
        // per iter; not the bottleneck per LDSM count)
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

                __half *row_lo_ptr = warp_weight + row_lo * W_STRIDE + tile_col_base;
                __half *row_hi_ptr = warp_weight + row_hi * W_STRIDE + tile_col_base;
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

        __syncthreads();

        // Phase D: PV WMMA — v_frag stride changes from D_HEAD to D_STRIDE
        if (valid_warp) {
            #pragma unroll
            for (int n = 0; n < TILES_D; n++) {
                #pragma unroll
                for (int k = 0; k < TILES_Bc; k++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                    wmma::load_matrix_sync(w_frag,
                        warp_weight + k * WMMA_K, W_STRIDE);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        cur_V + k * WMMA_K * D_STRIDE + n * WMMA_N,
                        D_STRIDE);

                    wmma::mma_sync(pv_accum[n], w_frag, v_frag, pv_accum[n]);
                }
            }
        }

        __syncthreads();
    }

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

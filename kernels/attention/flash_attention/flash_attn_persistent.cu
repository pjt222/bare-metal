/*
 * flash_attn_persistent.cu — Persistent-grid Flash Attention
 *
 * Same algorithm as flash_attn_br16 but with a persistent kernel grid:
 * instead of launching (q_tiles × heads × batch) blocks, we launch
 * exactly num_sms × 2 blocks and each block loops, grabbing work tiles
 * via atomicAdd on a global counter.
 *
 * Motivation: at small batch sizes (batch=1, heads=8, seq=256), the
 * standard grid has only 32 blocks for 48 SMs — 16 SMs idle, each busy
 * SM has 1 block = 4 warps (below the 8-warp latency-hiding threshold).
 * The persistent grid fills all SMs from the start.
 *
 * Work tile = one Q-tile (64 rows) over all KV tiles for one (head, batch).
 * Total tiles = q_tiles_per_seq × num_heads × batch_size.
 *
 * Grid:  (num_sms * 2, 1, 1)  — 2 blocks/SM at 48 KB smem
 * Block: (128, 1, 1) = 4 warps
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_persistent.sm_86.cubin flash_attn_persistent.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WARP_SIZE   32
#define NUM_WARPS   4
#define D_HEAD      64
#define Br_WARP     16
#define Br_BLOCK    (NUM_WARPS * Br_WARP)      // 64
#define Bc          64
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define TILES_D     (D_HEAD / WMMA_K)          // 4
#define TILES_Bc    (Bc / WMMA_N)              // 4

#define LOG2E   1.4426950408889634f

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void flash_attn_persistent(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    int   batch_size,
    float scale,
    int   total_tiles,       // q_tiles_per_seq * num_heads * batch_size
    int   q_tiles_per_seq,   // ceil(seq_len / Br_BLOCK)
    int * __restrict__ tile_counter  // global atomic counter, must be zeroed before launch
) {
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + 1 * Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));
    float  *smem_pv   = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half)
                                           + Br_BLOCK * Bc * sizeof(float));

    int global_thread = threadIdx.x;
    int warp_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    size_t head_stride  = (size_t)seq_len * D_HEAD;
    size_t batch_stride = (size_t)num_heads * head_stride;

    // ====================================================================
    // Persistent work-stealing loop
    // ====================================================================
    while (true) {
        // --- Grab next work tile (one thread does atomic, broadcast via smem) ---
        __shared__ int shared_tile_idx;
        if (global_thread == 0) {
            shared_tile_idx = atomicAdd(tile_counter, 1);
        }
        __syncthreads();
        int tile_idx = shared_tile_idx;

        if (tile_idx >= total_tiles) break;

        // --- Decode tile → (batch, head, q_tile) ---
        int batch_idx = tile_idx / (num_heads * q_tiles_per_seq);
        int remainder = tile_idx % (num_heads * q_tiles_per_seq);
        int head_idx  = remainder / q_tiles_per_seq;
        int q_tile    = remainder % q_tiles_per_seq;

        int block_q_base = q_tile * Br_BLOCK;
        int warp_q_base  = block_q_base + warp_id * Br_WARP;

        size_t base_offset = (size_t)batch_idx * batch_stride + (size_t)head_idx * head_stride;
        const __half *Q_head = Q + base_offset;
        const __half *K_head = K + base_offset;
        const __half *V_head = V + base_offset;
        float        *O_head = O + base_offset;

        bool valid_warp = (warp_q_base < seq_len);

        // Per-warp section pointers into smem
        float *warp_work = smem_work + warp_id * Br_WARP * Bc;
        float *warp_pv   = smem_pv   + warp_id * Br_WARP * D_HEAD;

        // --- Initialize smem_pv to 0 ---
        for (int idx = global_thread; idx < Br_BLOCK * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
            smem_pv[idx] = 0.0f;
        }

        // --- Initialize per-warp running softmax state ---
        float running_max[Br_WARP];
        float running_sum[Br_WARP];
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            running_max[row] = -3.402823466e+38f;
            running_sum[row] = 0.0f;
        }

        __syncthreads();

        // ==============================================================
        // Main KV tile loop (identical to flash_attn_br16)
        // ==============================================================
        for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

            // Phase A: Load K_tile + V_tile
            for (int idx = global_thread; idx < Bc * D_HEAD; idx += NUM_WARPS * WARP_SIZE) {
                int kv_row    = idx / D_HEAD;
                int d_col     = idx % D_HEAD;
                int kv_global = kv_base + kv_row;
                K_tile[idx] = (kv_global < seq_len)
                    ? K_head[(size_t)kv_global * D_HEAD + d_col]
                    : __float2half(0.0f);
                V_tile[idx] = (kv_global < seq_len)
                    ? V_head[(size_t)kv_global * D_HEAD + d_col]
                    : __float2half(0.0f);
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
                            K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                            D_HEAD);
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
                        score_frag[n], Bc, wmma::mem_row_major);
                }
            }

            __syncthreads();

            // Phase C: Online softmax + PV WMMA
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

                // Phase D: PV WMMA
                __half *weight_ptr = (__half*)warp_work;

                #pragma unroll
                for (int n = 0; n < TILES_D; n++) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum;
                    wmma::load_matrix_sync(
                        pv_accum, warp_pv + n * WMMA_N, D_HEAD, wmma::mem_row_major);

                    #pragma unroll
                    for (int k = 0; k < TILES_Bc; k++) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> w_frag;
                        wmma::load_matrix_sync(w_frag, weight_ptr + k * WMMA_K, Bc);

                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
                        wmma::load_matrix_sync(v_frag,
                            V_tile + k * WMMA_K * D_HEAD + n * WMMA_N, D_HEAD);

                        wmma::mma_sync(pv_accum, w_frag, v_frag, pv_accum);
                    }

                    wmma::store_matrix_sync(
                        warp_pv + n * WMMA_N, pv_accum, D_HEAD, wmma::mem_row_major);
                }
            }

            __syncthreads();
        }

        // ==============================================================
        // Finalize: normalize and store output
        // ==============================================================
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

        __syncthreads();  // ensure output writes complete before smem reuse
    }
}

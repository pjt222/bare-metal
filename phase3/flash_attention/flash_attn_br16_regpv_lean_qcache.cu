/*
 * flash_attn_br16_regpv_lean_qcache.cu — lean state + Q register cache
 *
 * Variant of flash_attn_br16_regpv.cu with reduced per-thread register footprint.
 *
 * Observation:
 *   In the baseline regpv kernel, running_max[16] and running_sum[16] are
 *   declared as per-thread arrays. Because Phase C softmax computes each row's
 *   max/sum via warp-wide shfl_xor_sync reductions, ALL 32 lanes hold
 *   broadcast-identical copies of running_max[r] and running_sum[r]. That is
 *   16+16 = 32 redundant registers per thread.
 *
 * Restructure:
 *   In the WMMA fragment layout (sm_86), lane L "owns" 2 specific rows of the
 *   16x16 score tile:
 *     row_lo = (L >> 2)         (rows 0..7)
 *     row_hi = (L >> 2) + 8     (rows 8..15)
 *   So each lane only needs to persist the running_max/sum for its 2 owned rows
 *   (4 floats total). When Phase C iterates over row=0..15, the row's current
 *   running stat is fetched from the owning lane via __shfl_sync.
 *
 * Register savings:
 *   running_max:    16 → 2 (lo + hi)
 *   running_sum:    16 → 2 (lo + hi)
 *   total saving:   ~28-32 regs/thread
 *
 * Same shared memory layout as regpv (32 KB), same algorithm, same correctness.
 * Same 3 blocks/SM occupancy; freed regs become headroom for downstream
 * optimizations (Q caching, padded smem, etc.).
 *
 * Plus: Q fragments hoisted out of the KV loop (loaded once per kernel,
 * held in registers across all KV iterations). Eliminates redundant L2
 * traffic: was 16x reload at seq=1024.
 *
 * Q frag storage: TILES_D=4 fragments × 4 uint32/lane = 16 regs/thread.
 * With lean baseline at 132 regs, adds to ~148 regs/thread — still fits 3 blocks/SM
 * (148 * 128 * 3 = 56832 < 65536).
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o flash_br16_regpv_lean_qcache.sm_86.cubin flash_attn_br16_regpv_lean_qcache.cu
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

extern "C" __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 3)
void flash_attn_br16_regpv_lean_qcache(
    const __half * __restrict__ Q,
    const __half * __restrict__ K,
    const __half * __restrict__ V,
    float        * __restrict__ O,
    int   seq_len,
    int   num_heads,
    float scale
) {
    extern __shared__ char smem_raw[];

    __half *K_tile    = (__half*)(smem_raw);
    __half *V_tile    = (__half*)(smem_raw + Bc * D_HEAD * sizeof(__half));
    float  *smem_work = (float *)(smem_raw + 2 * Bc * D_HEAD * sizeof(__half));

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

    float *warp_work = smem_work + warp_id * Br_WARP * Bc;

    int groupID = lane >> 2;        // 0..7  (lane owns row groupID and groupID+8)

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_accum[TILES_D];
    #pragma unroll
    for (int n = 0; n < TILES_D; n++) wmma::fill_fragment(pv_accum[n], 0.0f);

    // Lean per-thread state: only 2 owned rows × 2 stats = 4 floats
    float my_max_lo = -3.402823466e+38f;
    float my_max_hi = -3.402823466e+38f;
    float my_sum_lo = 0.0f;
    float my_sum_hi = 0.0f;

    // Q register cache — loaded once per kernel, reused across all KV iterations.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag[TILES_D];
    if (valid_warp) {
        #pragma unroll
        for (int dk = 0; dk < TILES_D; dk++) {
            wmma::load_matrix_sync(q_frag[dk],
                Q_head + (size_t)warp_q_base * D_HEAD + dk * WMMA_K,
                D_HEAD);
        }
    }

    __syncthreads();

    for (int kv_base = 0; kv_base < seq_len; kv_base += Bc) {

        // Phase A: load K, V
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

        // Phase B: QK^T via WMMA
        if (valid_warp) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag[TILES_Bc];
            #pragma unroll
            for (int n = 0; n < TILES_Bc; n++) wmma::fill_fragment(score_frag[n], 0.0f);

            // Q is cached in registers (q_frag[]); no global reload.
            #pragma unroll
            for (int dk = 0; dk < TILES_D; dk++) {
                #pragma unroll
                for (int n = 0; n < TILES_Bc; n++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(k_frag,
                        K_tile + n * WMMA_N * D_HEAD + dk * WMMA_K,
                        D_HEAD);
                    wmma::mma_sync(score_frag[n], q_frag[dk], k_frag, score_frag[n]);
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
                    score_frag[n],
                    Bc,
                    wmma::mem_row_major);
            }
        }

        __syncthreads();

        // Phase C: online softmax with lean per-thread state
        //
        // For each row r in 0..15:
        //   1. Owning lane(s) hold this row's running_max in my_max_lo (r<8)
        //      or my_max_hi (r>=8). All groupID==r%8 lanes (4 lanes) hold copy.
        //   2. Broadcast cur_running_max from lane (r%8)*4 to all 32 lanes via shfl.
        //   3. Reduce partial_max across 32 lanes via shfl_xor (existing logic).
        //   4. Compute new_max identically on all 32 lanes.
        //   5. Update only the owning lane's storage.
        //
        // After the loop, my_max_lo/my_max_hi/my_sum_lo/my_sum_hi reflect
        // post-softmax state for this lane's owned rows.
        //
        // For applying rescale to pv_accum: this lane needs s_lo (rescale for
        // row_lo) and s_hi (rescale for row_hi), which are produced naturally
        // when row==row_lo and row==row_hi iterations write to my_*.
        // We capture these into local s_lo / s_hi.
        if (valid_warp) {
            float s_lo = 1.0f, s_hi = 1.0f;

            #pragma unroll
            for (int row = 0; row < Br_WARP; row++) {
                float *score_row = warp_work + row * Bc;

                // Reduce partial_max across 32 lanes (Bc=64 cols → 2 per lane)
                float partial_max = fmaxf(score_row[lane], score_row[lane + WARP_SIZE]);
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_max = fmaxf(partial_max, __shfl_xor_sync(0xFFFFFFFF, partial_max, off));

                // Fetch this row's current running_max from owning lane
                int src_lane = (row < 8) ? (row << 2) : ((row - 8) << 2);
                float my_owned = (row < 8) ? my_max_lo : my_max_hi;
                float cur_running_max = __shfl_sync(0xFFFFFFFF, my_owned, src_lane);

                float new_max        = fmaxf(cur_running_max, partial_max);
                float rescale_factor = exp2f((cur_running_max - new_max) * LOG2E);

                // Update owning lane's storage. Capture s_lo/s_hi for pv_accum apply.
                if (row < 8) {
                    if (groupID == row) {
                        my_max_lo = new_max;
                        s_lo = rescale_factor;
                    }
                } else {
                    if (groupID == row - 8) {
                        my_max_hi = new_max;
                        s_hi = rescale_factor;
                    }
                }

                // Compute exp weights (FP16 overlay for Phase D)
                float w_lo = exp2f((score_row[lane]             - new_max) * LOG2E);
                float w_hi = exp2f((score_row[lane + WARP_SIZE] - new_max) * LOG2E);

                __half *weight_row = (__half*)warp_work + row * Bc;
                weight_row[lane]             = __float2half(w_lo);
                weight_row[lane + WARP_SIZE] = __float2half(w_hi);

                // Reduce partial_sum across 32 lanes
                float partial_sum = w_lo + w_hi;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, off);

                // Update owning lane's sum: rescale old + add new partial_sum
                if (row < 8) {
                    if (groupID == row) {
                        my_sum_lo = my_sum_lo * rescale_factor + partial_sum;
                    }
                } else {
                    if (groupID == row - 8) {
                        my_sum_hi = my_sum_hi * rescale_factor + partial_sum;
                    }
                }
            }

            // Apply rescale to register-resident pv_accum.
            // s_lo and s_hi were set during the row loop iterations row==row_lo
            // and row==row_hi respectively. They are the per-thread rescale
            // factors for this lane's two owned rows.
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

            // Phase D: PV WMMA
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

        __syncthreads();
    }

    // Finalize: store pv_accum, normalize using lane-owned my_sum_lo/hi
    if (valid_warp) {
        #pragma unroll
        for (int n = 0; n < TILES_D; n++) {
            wmma::store_matrix_sync(
                warp_work + n * WMMA_N,
                pv_accum[n],
                Bc,
                wmma::mem_row_major);
        }
    }

    __syncthreads();

    // For final output, each row needs 1/running_sum[row]. Each lane owns 2 rows.
    // Broadcast pattern: lane r%8*4 holds my_sum_lo for row<8 (or my_sum_hi for row>=8).
    if (valid_warp) {
        #pragma unroll
        for (int row = 0; row < Br_WARP; row++) {
            int global_q = warp_q_base + row;
            if (global_q >= seq_len) break;

            int src_lane = (row < 8) ? (row << 2) : ((row - 8) << 2);
            float my_owned_sum = (row < 8) ? my_sum_lo : my_sum_hi;
            float row_sum = __shfl_sync(0xFFFFFFFF, my_owned_sum, src_lane);

            float rcp_sum = __frcp_rn(row_sum);
            float *src = warp_work + row * Bc;

            O_head[(size_t)global_q * D_HEAD + lane]             = src[lane]             * rcp_sum;
            O_head[(size_t)global_q * D_HEAD + lane + WARP_SIZE] = src[lane + WARP_SIZE] * rcp_sum;
        }
    }
}

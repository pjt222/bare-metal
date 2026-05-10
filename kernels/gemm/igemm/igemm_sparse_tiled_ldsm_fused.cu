/*
 * igemm_sparse_tiled_ldsm_fused.cu — Fused B-load: no temp buffer, no SWIZZLE_B.
 */

#include <cuda_pipeline.h>
#include <cstdint>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 32
#define BM 128
#define BN 128
#define BK 64
#define BK_STRIDE 68
#define BK_COMP (BK / 2)
#define STRIDE_A  (BK_COMP + 16)
#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)
#define WARPS_Y 4
#define WARPS_X 4
#define WARP_TILES_M 2
#define WARP_TILES_N 2

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void igemm_sparse_tiled(
    const int8_t     * __restrict__ A_compressed,
    const int8_t     * __restrict__ B,
    float            * __restrict__ C,
    const uint32_t   * __restrict__ metadata,
    int M, int N, int K,
    float scale_a, float scale_b
) {
    __shared__ __align__(16) int8_t smem_a[2][BM * STRIDE_A];
    __shared__ __align__(16) int8_t smem_b_reformat[2][BN * BK_STRIDE];
    __shared__ __align__(16) uint32_t smem_meta[2][128];

    int thread_id = threadIdx.x;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;
    int wy        = warp_id / WARPS_X;
    int wx        = warp_id % WARPS_X;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int K_stored      = K / 2;
    int K_steps_total = K / WMMA_K;
    int gid      = lane >> 2;
    int tid_frag = lane & 3;

    int32_t acc_left[WARP_TILES_M][WARP_TILES_N][4];
    int32_t acc_right[WARP_TILES_M][WARP_TILES_N][4];
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            acc_left[wi][wj][0] = 0; acc_left[wi][wj][1] = 0;
            acc_left[wi][wj][2] = 0; acc_left[wi][wj][3] = 0;
            acc_right[wi][wj][0] = 0; acc_right[wi][wj][1] = 0;
            acc_right[wi][wj][2] = 0; acc_right[wi][wj][3] = 0;
        }
    }

    int num_tiles = (K + BK - 1) / BK;

    #define LOAD_A_TILE(buf, kb)                                                     \
    {                                                                                 \
        int _kc = (kb) / 2;                                                          \
        int _f  = thread_id;                                                         \
        if (_f < (BM * BK_COMP) / 16) {                                             \
            int _be = _f * 16;                                                       \
            int _r  = _be / BK_COMP;                                                 \
            int _c  = _be % BK_COMP;                                                 \
            int _so = _r * STRIDE_A + _c;                                            \
            int _gr = block_row + _r;                                                \
            int _gc = _kc + _c;                                                      \
            if (_gr < M && _gc + 15 < K_stored) {                                    \
                __pipeline_memcpy_async(&smem_a[buf][_so],                           \
                    &A_compressed[(size_t)_gr * K_stored + _gc], 16);                \
            } else {                                                                 \
                for (int _b = 0; _b < 16; _b++)                                      \
                    smem_a[buf][_so + _b] = (_gr < M && _gc + _b < K_stored)         \
                        ? A_compressed[(size_t)_gr * K_stored + _gc + _b] : 0;       \
            }                                                                        \
        }                                                                            \
    }

    #define LOAD_B_TILE(buf, kb)                                                     \
    {                                                                                 \
        int _f = thread_id;                                                          \
        int _r   = _f / (BN / 16);                                                  \
        int _c   = (_f % (BN / 16)) * 16;                                           \
        int _gr  = (kb) + _r;                                                        \
        int _gc  = block_col + _c;                                                   \
        for (int _b = 0; _b < 16; _b++) {                                           \
            int _gn = _gc + _b;                                                      \
            int8_t _v = (_gr < K && _gn < N)                                         \
                ? B[(size_t)_gr * N + _gn] : 0;                                      \
            smem_b_reformat[buf][(_gn - block_col) * BK_STRIDE + _gr - (kb)] = _v;                \
        }                                                                            \
    }

    #define LOAD_META_TILE(buf, ti)                                                  \
    {                                                                                 \
        int _gk = (ti) * (BK / WMMA_K);                                            \
        int _f = thread_id;                                                          \
        if (_f < 128) {                                                            \
            int _mo = _f / 16;                                                       \
            int _koff = (_f % 16) / 8;                                              \
            int _g = _f % 8;                                                        \
            int _m = block_row / WMMA_M + _mo;                                       \
            int _ki = _gk + _koff;                                                   \
            smem_meta[buf][_f] = metadata[(size_t)_m * K_steps_total * 8 + (size_t)_ki * 8 + _g]; \
        }                                                                            \
    }

    #define COMPUTE_TILE(buf, ti)                                                    \
    _Pragma("unroll")                                                                \
    for (int k_step = 0; k_step < BK; k_step += WMMA_K) {                          \
        int k_comp = k_step / 2;                                                     \
        int _gki = (ti) * (BK / WMMA_K) + k_step / WMMA_K;                         \
        int kg0 = k_step / 4 + tid_frag;                                             \
        _Pragma("unroll")                                                            \
        for (int wi = 0; wi < WARP_TILES_M; wi++) {                                 \
            int a_row = wy * 32 + wi * WMMA_M;                                       \
            uint32_t meta = smem_meta[buf][(wy * 2 + wi) * 16 + (_gki % 2) * 8 + gid]; \
            uint32_t fa0, fa1;                                                       \
            {                                                                        \
                uint32_t pa = __cvta_generic_to_shared(                              \
                    &smem_a[buf][(a_row + (lane & 15)) * STRIDE_A + k_comp]);       \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" \
                    : "=r"(fa0), "=r"(fa1) : "r"(pa));                             \
            }                                                                        \
            _Pragma("unroll")                                                        \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                             \
                int b_col = wx * 32 + wj * WMMA_N;                                   \
                uint32_t fb_l0, fb_l1, fb_r0, fb_r1;                                 \
                {                                                                    \
                    int _brow0 = kg0 * 4;                                            \
                    int _brow1 = _brow0 + 16;                                        \
                    uint32_t al = __cvta_generic_to_shared(                          \
                        &smem_b_reformat[buf][(b_col + gid) * BK_STRIDE + _brow0]); \
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(fb_l0) : "r"(al)); \
                    al = __cvta_generic_to_shared(                                   \
                        &smem_b_reformat[buf][(b_col + gid) * BK_STRIDE + _brow1]); \
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(fb_l1) : "r"(al)); \
                    al = __cvta_generic_to_shared(                                   \
                        &smem_b_reformat[buf][(b_col + 8 + gid) * BK_STRIDE + _brow0]); \
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(fb_r0) : "r"(al)); \
                    al = __cvta_generic_to_shared(                                   \
                        &smem_b_reformat[buf][(b_col + 8 + gid) * BK_STRIDE + _brow1]); \
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(fb_r1) : "r"(al)); \
                }                                                                    \
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;"   \
                    : "=r"(acc_left[wi][wj][0]), "=r"(acc_left[wi][wj][1]),         \
                      "=r"(acc_left[wi][wj][2]), "=r"(acc_left[wi][wj][3])         \
                    : "r"(fa0), "r"(fa1), "r"(fb_l0), "r"(fb_l1),                   \
                      "r"(acc_left[wi][wj][0]), "r"(acc_left[wi][wj][1]),            \
                      "r"(acc_left[wi][wj][2]), "r"(acc_left[wi][wj][3]),            \
                      "r"(meta));                                                     \
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;"   \
                    : "=r"(acc_right[wi][wj][0]), "=r"(acc_right[wi][wj][1]),       \
                      "=r"(acc_right[wi][wj][2]), "=r"(acc_right[wi][wj][3])      \
                    : "r"(fa0), "r"(fa1), "r"(fb_r0), "r"(fb_r1),                   \
                      "r"(acc_right[wi][wj][0]), "r"(acc_right[wi][wj][1]),          \
                      "r"(acc_right[wi][wj][2]), "r"(acc_right[wi][wj][3]),          \
                      "r"(meta));                                                     \
            }                                                                        \
        }                                                                            \
    }

    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    LOAD_META_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k = (tile + 1) * BK;
        int cur = tile & 1;
        int nxt = 1 - cur;
        LOAD_A_TILE(nxt, next_k);
        LOAD_B_TILE(nxt, next_k);
        LOAD_META_TILE(nxt, tile + 1);
        __pipeline_commit();
        COMPUTE_TILE(cur, tile);
        __pipeline_wait_prior(0);
        __syncthreads();
    }
    COMPUTE_TILE((num_tiles - 1) & 1, num_tiles - 1);

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef LOAD_META_TILE
    #undef COMPUTE_TILE

    float ds = scale_a * scale_b;
    int sc0 = tid_frag * 2;
    int sc1 = sc0 + 1;
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int crow = block_row + wy * 32 + wi * WMMA_M;
            int ccol = block_col + wx * 32 + wj * WMMA_N;
            int rlo = crow + gid;
            int rhi = crow + gid + 8;
            if (rlo < M) {
                if (ccol + sc0 < N) C[(size_t)rlo * N + ccol + sc0] = (float)acc_left[wi][wj][0] * ds;
                if (ccol + sc1 < N) C[(size_t)rlo * N + ccol + sc1] = (float)acc_left[wi][wj][1] * ds;
            }
            if (rhi < M) {
                if (ccol + sc0 < N) C[(size_t)rhi * N + ccol + sc0] = (float)acc_left[wi][wj][2] * ds;
                if (ccol + sc1 < N) C[(size_t)rhi * N + ccol + sc1] = (float)acc_left[wi][wj][3] * ds;
            }
            if (rlo < M) {
                if (ccol + 8 + sc0 < N) C[(size_t)rlo * N + ccol + 8 + sc0] = (float)acc_right[wi][wj][0] * ds;
                if (ccol + 8 + sc1 < N) C[(size_t)rlo * N + ccol + 8 + sc1] = (float)acc_right[wi][wj][1] * ds;
            }
            if (rhi < M) {
                if (ccol + 8 + sc0 < N) C[(size_t)rhi * N + ccol + 8 + sc0] = (float)acc_right[wi][wj][2] * ds;
                if (ccol + 8 + sc1 < N) C[(size_t)rhi * N + ccol + 8 + sc1] = (float)acc_right[wi][wj][3] * ds;
            }
        }
    }
}

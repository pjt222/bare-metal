/*
 * igemm_sparse_tiled_persistent.cu — Persistent grid Sparse INT8 GEMM
 * Blocks dynamically claim output tiles via atomic counter.
 */

#include <cuda_pipeline.h>
#include <cstdint>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 32
#define BM 128
#define BN 128
#define BK 64
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
void igemm_sparse_tiled_persistent(
    const int8_t     * __restrict__ A_compressed,
    const int8_t     * __restrict__ B,
    float            * __restrict__ C,
    const uint32_t   * __restrict__ metadata,
    int M, int N, int K,
    float scale_a, float scale_b,
    unsigned int     * __restrict__ tile_counter
) {
    __shared__ __align__(16) int8_t smem_a[2][BM * STRIDE_A];
    __shared__ __align__(16) int8_t smem_b[2][BK * (BN + 16)];
    __shared__ __align__(16) uint32_t smem_meta[2][128];

    int thread_id = threadIdx.x;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;
    int wy        = warp_id / WARPS_X;
    int wx        = warp_id % WARPS_X;
    int K_stored  = K / 2;
    int K_steps_total = K / WMMA_K;
    int gid       = lane >> 2;
    int tid_frag  = lane & 3;
    int num_tiles_k = (K + BK - 1) / BK;
    int num_tiles_m = (M + BM - 1) / BM;
    int num_tiles_n = (N + BN - 1) / BN;
    int total_tiles = num_tiles_m * num_tiles_n;

    int32_t acc_left[WARP_TILES_M][WARP_TILES_N][4];
    int32_t acc_right[WARP_TILES_M][WARP_TILES_N][4];

    #define INIT_ACC()                                                          \
        for (int _wi = 0; _wi < WARP_TILES_M; _wi++) {                         \
            for (int _wj = 0; _wj < WARP_TILES_N; _wj++) {                     \
                acc_left[_wi][_wj][0]=0; acc_left[_wi][_wj][1]=0;               \
                acc_left[_wi][_wj][2]=0; acc_left[_wi][_wj][3]=0;               \
                acc_right[_wi][_wj][0]=0; acc_right[_wi][_wj][1]=0;             \
                acc_right[_wi][_wj][2]=0; acc_right[_wi][_wj][3]=0;             \
            }                                                                   \
        }

    #define LOAD_A_TILE(buf, kb, brow)                                           \
    {                                                                             \
        int _kc = (kb) / 2;                                                      \
        int _f  = thread_id;                                                     \
        if (_f < (BM * BK_COMP) / 16) {                                         \
            int _be = _f * 16;                                                   \
            int _r  = _be / BK_COMP;                                             \
            int _c  = _be % BK_COMP;                                             \
            int _so = _r * STRIDE_A + _c;                                        \
            int _gr = (brow) + _r;                                               \
            int _gc = _kc + _c;                                                  \
            if (_gr < M && _gc + 15 < K_stored) {                                \
                __pipeline_memcpy_async(&smem_a[buf][_so],                       \
                    &A_compressed[(size_t)_gr * K_stored + _gc], 16);            \
            } else {                                                             \
                for (int _b = 0; _b < 16; _b++)                                  \
                    smem_a[buf][_so + _b] = (_gr < M && _gc + _b < K_stored)     \
                        ? A_compressed[(size_t)_gr * K_stored + _gc + _b] : 0;   \
            }                                                                    \
        }                                                                        \
    }

    #define LOAD_B_TILE(buf, kb, bcol)                                           \
    {                                                                             \
        int _f  = thread_id;                                                     \
        int _r  = _f / (BN / 16);                                              \
        int _c  = (_f % (BN / 16)) * 16;                                       \
        int _so = _r * (BN + 16) + _c;                                         \
        int _gr = (kb) + _r;                                                     \
        int _gc = (bcol) + _c;                                                   \
        if (_gr < K && _gc + 15 < N) {                                          \
            __pipeline_memcpy_async(&smem_b[buf][_so],                           \
                &B[(size_t)_gr * N + _gc], 16);                                  \
        } else {                                                                 \
            for (int _b = 0; _b < 16; _b++)                                      \
                smem_b[buf][_so + _b] = (_gr < K && _gc + _b < N)                \
                    ? B[(size_t)_gr * N + _gc + _b] : 0;                         \
        }                                                                        \
    }

    #define LOAD_META_TILE(buf, ti, brow)                                        \
    {                                                                             \
        int _gk = (ti) * (BK / WMMA_K);                                        \
        int _f  = thread_id;                                                     \
        if (_f < 128) {                                                        \
            int _mo   = _f / 16;                                                 \
            int _koff = (_f % 16) / 8;                                          \
            int _g    = _f % 8;                                                 \
            int _m    = (brow) / WMMA_M + _mo;                                  \
            int _ki   = _gk + _koff;                                           \
            smem_meta[buf][_f] = metadata[(size_t)_m * K_steps_total * 8       \
                                         + (size_t)_ki * 8 + _g];             \
        }                                                                        \
    }

    #define COMPUTE_TILE(buf, ti)                                                \
    _Pragma("unroll")                                                            \
    for (int k_step = 0; k_step < BK; k_step += WMMA_K) {                      \
        int k_comp = k_step / 2;                                                 \
        int _gki   = (ti) * (BK / WMMA_K) + k_step / WMMA_K;                   \
        int _k0    = k_step + tid_frag * 4;                                      \
        int _k1    = k_step + tid_frag * 4 + 16;                                 \
        _Pragma("unroll")                                                        \
        for (int wi = 0; wi < WARP_TILES_M; wi++) {                             \
            int a_base_row = wy * 32 + wi * WMMA_M;                             \
            uint32_t meta = smem_meta[buf][(wy * 2 + wi) * 16 + (_gki % 2) * 8 + gid]; \
            uint32_t fa0, fa1;                                                   \
            {                                                                    \
                uint32_t _pa = __cvta_generic_to_shared(                        \
                    &smem_a[buf][(a_base_row + (lane & 15)) * STRIDE_A + k_comp]); \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" \
                    : "=r"(fa0), "=r"(fa1) : "r"(_pa));                        \
            }                                                                    \
            _Pragma("unroll")                                                    \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                         \
                int b_base_col  = wx * 32 + wj * WMMA_N;                        \
                int _ncol_left  = b_base_col + gid;                             \
                int _ncol_right = b_base_col + 8 + gid;                         \
                uint32_t fb_left0, fb_left1, fb_right0, fb_right1;              \
                {                                                                \
                    const int8_t *_p = &smem_b[buf][_k0 * (BN + 16) + _ncol_left];  \
                    fb_left0 = (uint32_t)(uint8_t)_p[0]                         \
                             | ((uint32_t)(uint8_t)_p[(BN + 16)]     << 8)      \
                             | ((uint32_t)(uint8_t)_p[(BN + 16) * 2] << 16)     \
                             | ((uint32_t)(uint8_t)_p[(BN + 16) * 3] << 24);    \
                    const int8_t *_q = &smem_b[buf][_k1 * (BN + 16) + _ncol_left];  \
                    fb_left1 = (uint32_t)(uint8_t)_q[0]                         \
                             | ((uint32_t)(uint8_t)_q[(BN + 16)]     << 8)      \
                             | ((uint32_t)(uint8_t)_q[(BN + 16) * 2] << 16)     \
                             | ((uint32_t)(uint8_t)_q[(BN + 16) * 3] << 24);    \
                }                                                                \
                {                                                                \
                    const int8_t *_p = &smem_b[buf][_k0 * (BN + 16) + _ncol_right]; \
                    fb_right0 = (uint32_t)(uint8_t)_p[0]                        \
                              | ((uint32_t)(uint8_t)_p[(BN + 16)]     << 8)     \
                              | ((uint32_t)(uint8_t)_p[(BN + 16) * 2] << 16)    \
                              | ((uint32_t)(uint8_t)_p[(BN + 16) * 3] << 24);   \
                    const int8_t *_q = &smem_b[buf][_k1 * (BN + 16) + _ncol_right]; \
                    fb_right1 = (uint32_t)(uint8_t)_q[0]                        \
                              | ((uint32_t)(uint8_t)_q[(BN + 16)]     << 8)     \
                              | ((uint32_t)(uint8_t)_q[(BN + 16) * 2] << 16)    \
                              | ((uint32_t)(uint8_t)_q[(BN + 16) * 3] << 24);   \
                }                                                                \
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;" \
                    : "=r"(acc_left[wi][wj][0]), "=r"(acc_left[wi][wj][1]),      \
                      "=r"(acc_left[wi][wj][2]), "=r"(acc_left[wi][wj][3])     \
                    : "r"(fa0), "r"(fa1), "r"(fb_left0), "r"(fb_left1),          \
                      "r"(acc_left[wi][wj][0]), "r"(acc_left[wi][wj][1]),        \
                      "r"(acc_left[wi][wj][2]), "r"(acc_left[wi][wj][3]),        \
                      "r"(meta));                                                 \
                asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;" \
                    : "=r"(acc_right[wi][wj][0]), "=r"(acc_right[wi][wj][1]),    \
                      "=r"(acc_right[wi][wj][2]), "=r"(acc_right[wi][wj][3])   \
                    : "r"(fa0), "r"(fa1), "r"(fb_right0), "r"(fb_right1),        \
                      "r"(acc_right[wi][wj][0]), "r"(acc_right[wi][wj][1]),      \
                      "r"(acc_right[wi][wj][2]), "r"(acc_right[wi][wj][3]),      \
                      "r"(meta));                                                 \
            }                                                                    \
        }                                                                        \
    }

    // --- Persistent grid: claim tiles ---
    __shared__ unsigned int s_tile_idx;
    while (true) {
        if (thread_id == 0)
            s_tile_idx = atomicAdd(tile_counter, 1);
        __syncthreads();
        if (s_tile_idx >= (unsigned int)total_tiles) break;

        int tile_m = s_tile_idx / num_tiles_n;
        int tile_n = s_tile_idx % num_tiles_n;
        int block_row = tile_m * BM;
        int block_col = tile_n * BN;

        INIT_ACC();

        LOAD_A_TILE(0, 0, block_row);
        LOAD_B_TILE(0, 0, block_col);
        LOAD_META_TILE(0, 0, block_row);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int tile = 0; tile < num_tiles_k - 1; tile++) {
            int next_k = (tile + 1) * BK;
            int cur = tile & 1;
            int nxt = 1 - cur;
            LOAD_A_TILE(nxt, next_k, block_row);
            LOAD_B_TILE(nxt, next_k, block_col);
            LOAD_META_TILE(nxt, tile + 1, block_row);
            __pipeline_commit();
            COMPUTE_TILE(cur, tile);
            __pipeline_wait_prior(0);
            __syncthreads();
        }
        COMPUTE_TILE((num_tiles_k - 1) & 1, num_tiles_k - 1);

        float ds = scale_a * scale_b;
        int sc0 = tid_frag * 2;
        int sc1 = sc0 + 1;
        for (int wi = 0; wi < WARP_TILES_M; wi++) {
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

    #undef INIT_ACC
    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef LOAD_META_TILE
    #undef COMPUTE_TILE
}

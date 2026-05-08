/*
 * hgemm_dispatch.cuh — choose the best HGEMM variant for a given (M, N, K).
 *
 * Header-only dispatcher. The decision matrix is built from measurements
 * in `bench_splitk.cu` (Observation EE):
 *
 *   - For "skinny" shapes (small M*N), K-split wins 1.7x to 4.6x.
 *   - For "square" shapes (M*N >= 1024^2), the standard kernel wins
 *     because atomicAdd cost + 1 block/SM overhead exceed the
 *     parallelism gain from K-split.
 *
 * The dispatcher launches one of:
 *   hgemm_16warp           — standard (one block per output tile,
 *                            2 blocks/SM with PAD_A=PAD_B=8)
 *   hgemm_16warp_splitk    — K-split with atomicAdd, dynamic smem 53 KB,
 *                            1 block/SM
 *
 * Caller responsibilities:
 *   - Both cubins loaded, function handles passed to dispatch_init().
 *   - For the splitk path, dispatcher will zero matrix_c via
 *     cuMemsetD8 before launch.
 */

#ifndef HGEMM_DISPATCH_CUH_
#define HGEMM_DISPATCH_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

namespace hgemm_dispatch {

struct Handles {
    CUfunction fn_standard;          // hgemm_16warp
    CUfunction fn_splitk;            // hgemm_16warp_splitk
    size_t     splitk_smem_bytes;    // dynamic smem for splitk (set via cuFuncSetAttribute)
};

enum class Variant : uint8_t {
    Standard,
    SplitK_2,
    SplitK_4,
    SplitK_8
};

inline const char* variant_name(Variant v) {
    switch (v) {
        case Variant::Standard:  return "standard";
        case Variant::SplitK_2:  return "splitk_2";
        case Variant::SplitK_4:  return "splitk_4";
        case Variant::SplitK_8:  return "splitk_8";
    }
    return "?";
}

// ----------------------------------------------------------------------
// Heuristic, derived from Observation EE (Phase 5 measurements):
//
//   M*N < 64 K   AND K >= 4096  ->  SplitK_8     (extreme skinny)
//   M*N < 256 K  AND K >= 4096  ->  SplitK_8     (skinny)
//   M*N < 1 M    AND K >= 4096  ->  SplitK_4 or 8 (mid skinny)
//   M*N >= 1 M                  ->  Standard     (square / large)
//
// Constants from the Observation EE table:
//   128x128x8192 = 16384  cells -> SplitK_8 (4.57x)
//   256x256x4096 = 65536  cells -> SplitK_8 (4.48x)
//   256x256x8192 = 65536  cells -> SplitK_4 (2.77x)
//   512x512x4096 = 262144 cells -> SplitK_8 (1.75x)
//   1024^3       = 1.05M  cells -> Standard wins
//
// We require K >= 4*BK so that K_split=8 still has at least 2 BK-sized
// tiles per slice. BK=32, so threshold K >= 256. In practice this holds
// for any non-degenerate problem.
// ----------------------------------------------------------------------

constexpr int  BK_SIZE        = 32;
constexpr long SQUARE_CUTOFF  = 1024L * 1024L;     // M*N >= 1M -> Standard
constexpr long SKINNY_CUTOFF  =  256L *  256L;     // M*N <  64K -> SplitK_8 (extreme)
constexpr int  MIN_K_SPLIT    =  256;              // require K >= MIN_K_SPLIT to consider any split

inline Variant pick_variant(int M, int N, int K) {
    long mn = (long)M * (long)N;

    if (mn >= SQUARE_CUTOFF || K < MIN_K_SPLIT) {
        return Variant::Standard;
    }

    int k_tiles = (K + BK_SIZE - 1) / BK_SIZE;

    // Choose split factor: prefer 8 if cleanly divisible, else 4, else 2.
    auto fits = [&](int s) { return s <= k_tiles && (k_tiles % s == 0); };

    if (mn <= SKINNY_CUTOFF) {                     // very skinny -> max parallelism
        if (fits(8)) return Variant::SplitK_8;
        if (fits(4)) return Variant::SplitK_4;
        if (fits(2)) return Variant::SplitK_2;
        return Variant::Standard;
    }

    // Mid-skinny: 256K <= M*N < 1M cells.
    // From the EE table 512x512x4096 (262144 cells, K=4096) chose split_8.
    // 256x256x8192 (65536 cells, K=8192) chose split_4 -- with very large K
    // a slightly smaller split balances atomicAdd cost.
    if (K >= 8192) {
        if (fits(4)) return Variant::SplitK_4;
        if (fits(2)) return Variant::SplitK_2;
    } else {
        if (fits(8)) return Variant::SplitK_8;
        if (fits(4)) return Variant::SplitK_4;
        if (fits(2)) return Variant::SplitK_2;
    }
    return Variant::Standard;
}

inline int variant_split_factor(Variant v) {
    switch (v) {
        case Variant::Standard:  return 1;
        case Variant::SplitK_2:  return 2;
        case Variant::SplitK_4:  return 4;
        case Variant::SplitK_8:  return 8;
    }
    return 1;
}

// ----------------------------------------------------------------------
// Launch a chosen variant. Returns the variant actually used.
// Caller passes already-loaded function handles; both must be valid.
// ----------------------------------------------------------------------
inline Variant launch(const Handles &h,
                      CUdeviceptr dA, CUdeviceptr dB, CUdeviceptr dC,
                      int M, int N, int K)
{
    constexpr int BM = 128, BN = 128;
    constexpr int BLOCK_THREADS = 512;

    Variant v = pick_variant(M, N, K);
    int split = variant_split_factor(v);
    int gx = N / BN, gy = M / BM;

    if (v == Variant::Standard) {
        void *args[] = { &dA, &dB, &dC, &M, &N, &K };
        CUresult rc = cuLaunchKernel(h.fn_standard,
            gx, gy, 1, BLOCK_THREADS, 1, 1, 0, 0, args, 0);
        if (rc != CUDA_SUCCESS) {
            const char *errstr = nullptr; cuGetErrorString(rc, &errstr);
            fprintf(stderr, "dispatch: standard launch failed: %s\n", errstr);
        }
    } else {
        // splitk requires zero'd C
        cuMemsetD8(dC, 0, (size_t)M * N * sizeof(float));
        void *args[] = { &dA, &dB, &dC, &M, &N, &K, &split };
        CUresult rc = cuLaunchKernel(h.fn_splitk,
            gx, gy, split, BLOCK_THREADS, 1, 1,
            (unsigned)h.splitk_smem_bytes, 0, args, 0);
        if (rc != CUDA_SUCCESS) {
            const char *errstr = nullptr; cuGetErrorString(rc, &errstr);
            fprintf(stderr, "dispatch: splitk launch failed: %s\n", errstr);
        }
    }
    return v;
}

}  // namespace hgemm_dispatch

#endif  // HGEMM_DISPATCH_CUH_

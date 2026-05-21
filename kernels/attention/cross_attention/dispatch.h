#pragma once
/*
 * dispatch.h — Cross-attention regime dispatch helper
 *
 * Selects the optimal cross-attention kernel variant based on problem size
 * (seq_q × seq_kv). The dispatch threshold is taken from the "Production
 * guidance" block of Obs X (docs/gpu_reflections.md, "Cross-attention
 * regime split"):
 *
 *   seq_q × seq_kv >= 200000  →  cross_attn_v2_pad   (bank-conflict-free, 27 KB)
 *   seq_q × seq_kv <  200000  →  cross_attn_br16      (baseline, 48 KB)
 *
 * The Obs X "Cross-attention regime split" table measures the padded
 * variant against the unpadded cross_attn_v2 (column `pad / v2`):
 *   256 × 77  (= 19 712, CLIP-77):    padding loses  (0.68× of v2)
 *   1024 × 256 (= 262 144, typical):  padding wins   (1.91× of v2)
 * Padding loses below the threshold and wins decisively above it.
 *
 * Note: Obs X's "Production guidance" block specifies a three-way
 * dispatch (baseline / v2 / v2_pad, with a second cut at 50 000). Issue
 * #103 scopes this helper to the two-way baseline / v2_pad split, so the
 * intermediate v2 tier is collapsed into the baseline branch here. This
 * is a scoping decision, not a measurement result — restore the v2 tier
 * from Obs X if a v2-only regime is needed.
 *
 * Usage:
 *   CrossAttnVariant v = cross_attn_pick(seq_q, seq_kv);
 *   CUfunction fn = load_kernel(v.cubin_path, v.symbol);
 *   cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)v.smem_bytes);
 *   cuLaunchKernel(fn, cross_attn_grid_x(seq_q), num_heads, batch,
 *                  CROSS_ATTN_BLOCK_THREADS, 1, 1, (unsigned)v.smem_bytes,
 *                  nullptr, args, nullptr);
 *
 * All tile constants below match the compiled .cu sources.  If the kernel
 * sources change their macros, update these constants and the smem formulas.
 *
 * Cross-references (docs/gpu_reflections.md, by observation heading):
 *   Obs P  — Flash Attention smem_work elimination: the structural win
 *   Obs X  — +8 padding generalizes; cross-attention regime split
 *   Obs JJ — Cymatic on Flash Attention K/V: cp.async structural limit
 */

#include <cstddef>

/* -------------------------------------------------------------------------
 * Compile-time tile constants — match macros in cross_attn*.cu sources.
 * -------------------------------------------------------------------------*/
static constexpr int    CROSS_ATTN_D_HEAD          = 64;   /* D_HEAD in all variants */
static constexpr int    CROSS_ATTN_BC              = 64;   /* Bc (KV tile columns)   */
static constexpr int    CROSS_ATTN_BR_BLOCK        = 64;   /* Br_BLOCK = 4 × 16      */
static constexpr int    CROSS_ATTN_D_PAD           = 8;    /* D_PAD in v2_pad        */
static constexpr int    CROSS_ATTN_D_STRIDE        = CROSS_ATTN_D_HEAD + CROSS_ATTN_D_PAD; /* 72 */
static constexpr int    CROSS_ATTN_BLOCK_THREADS   = 128;  /* 4 warps × 32 lanes     */
static constexpr size_t CROSS_ATTN_DISPATCH_THRESHOLD = 200000ULL; /* Obs X regime boundary */

/* -------------------------------------------------------------------------
 * Per-variant smem formulas.
 *
 * baseline (cross_attn_br16):
 *   K_tile:    Bc × D_HEAD     FP16  =  8 192 B
 *   V_tile:    Bc × D_HEAD     FP16  =  8 192 B
 *   smem_work: Br_BLOCK × Bc   FP32  = 16 384 B
 *   smem_pv:   Br_BLOCK × D_HEAD FP32= 16 384 B
 *   total                           = 49 152 B ≈ 48 KB
 *
 * v2_pad (cross_attn_v2_pad):
 *   K_tile:    Bc × D_STRIDE   FP16  =  9 216 B  (64×72×2)
 *   V_tile:    Bc × D_STRIDE   FP16  =  9 216 B
 *   weight_smem: Br_BLOCK × D_STRIDE FP16 = 9 216 B
 *     (D_STRIDE == W_STRIDE == 72 because D_PAD == W_PAD == 8 and D_HEAD == Bc == 64)
 *   total                           = 27 648 B ≈ 27 KB
 * -------------------------------------------------------------------------*/
static constexpr size_t CROSS_ATTN_SMEM_BASELINE =
    (size_t)2 * CROSS_ATTN_BC * CROSS_ATTN_D_HEAD * sizeof(short)       /* K + V tiles FP16 */
    + (size_t)CROSS_ATTN_BR_BLOCK * CROSS_ATTN_BC * sizeof(float)        /* smem_work  FP32  */
    + (size_t)CROSS_ATTN_BR_BLOCK * CROSS_ATTN_D_HEAD * sizeof(float);   /* smem_pv    FP32  */

static constexpr size_t CROSS_ATTN_SMEM_V2_PAD =
    (size_t)2 * CROSS_ATTN_BC * CROSS_ATTN_D_STRIDE * sizeof(short)     /* K + V tiles FP16 */
    + (size_t)CROSS_ATTN_BR_BLOCK * CROSS_ATTN_D_STRIDE * sizeof(short); /* weight_smem FP16 */
/* Note: in the v2_pad source, D_STRIDE == W_STRIDE == 72 because
 * D_PAD == W_PAD == 8 and D_HEAD == Bc == 64.  The formula above
 * assumes this coincidence; update both D_PAD and the W_STRIDE term
 * if the kernel macros diverge. */

/* -------------------------------------------------------------------------
 * CrossAttnVariant — all information a caller needs to launch one variant.
 * -------------------------------------------------------------------------*/
struct CrossAttnVariant {
    const char *cubin_path; /* cubin filename (relative — run from kernel dir) */
    const char *symbol;     /* kernel symbol name                               */
    size_t      smem_bytes; /* dynamic shared memory bytes per block            */
};

/* -------------------------------------------------------------------------
 * cross_attn_pick — regime dispatch
 *
 * Returns the CrossAttnVariant to use for a problem of size (seq_q, seq_kv).
 * The caller must:
 *   1. Load the cubin and obtain a CUfunction.
 *   2. Call cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes).
 *   3. Launch with grid (cross_attn_grid_x(seq_q), num_heads, batch),
 *      block (CROSS_ATTN_BLOCK_THREADS, 1, 1), and smem_bytes dynamic smem.
 *
 * The kernel arg signature for all variants:
 *   (const __half* Q, const __half* K, const __half* V, float* O,
 *    int seq_q, int seq_kv, int num_heads, float scale)
 * -------------------------------------------------------------------------*/
inline CrossAttnVariant cross_attn_pick(int seq_q, int seq_kv) {
    if ((size_t)seq_q * (size_t)seq_kv >= CROSS_ATTN_DISPATCH_THRESHOLD) {
        return { "cross_attn_v2_pad.sm_86.cubin", "cross_attn_v2_pad",
                 CROSS_ATTN_SMEM_V2_PAD };
    } else {
        return { "cross_attn.sm_86.cubin", "cross_attn_br16",
                 CROSS_ATTN_SMEM_BASELINE };
    }
}

/* -------------------------------------------------------------------------
 * cross_attn_grid_x — number of blocks in the Q dimension.
 *
 * Rounds seq_q up to the next multiple of Br_BLOCK so every block has a
 * full tile; the out-of-bounds rows are guarded inside the kernel.
 * -------------------------------------------------------------------------*/
inline int cross_attn_grid_x(int seq_q) {
    return (seq_q + CROSS_ATTN_BR_BLOCK - 1) / CROSS_ATTN_BR_BLOCK;
}

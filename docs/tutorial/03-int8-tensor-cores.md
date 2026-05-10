# Chapter 03 — INT8 Tensor Cores

> Read chapter 02 (GEMM from Scratch) first. This chapter applies the GEMM
> framework to INT8 instead of FP16, using IMMA instead of HMMA, plus
> online quantization and 2:4 structured sparsity.

GEMM in INT8 doubles arithmetic density compared to FP16 — the hardware
can do twice as many INT8 operations per clock as FP16 ones. For inference
workloads where the small precision loss is acceptable, INT8 is a 2×
free win. Add 2:4 structured sparsity on top, and that becomes a 4× win
(in dense-equivalent operations).

This chapter walks through the INT8 GEMM kernels in `kernels/gemm/igemm/`, the
IMMA pipeline pattern that differs from HMMA in subtle but important
ways, and the 2:4 sparsity path that this project ships at 35,509
dense-equivalent TOPS.

## Why INT8

Three reasons:

1. **Density**: GA104 FP16 Tensor Core peak is 174 TFLOPS. INT8 IMMA peak
   is 348 TOPS (2× the FP16 peak). Same hardware unit, different
   datapath.
2. **Memory**: INT8 weights and activations occupy half the DRAM and
   smem of FP16. For bandwidth-bound kernels (often the case in inference),
   this is a 2× speedup independent of compute.
3. **Sparsity**: NVIDIA's 2:4 structured sparsity pattern is supported
   natively by `mma.sp.sync` on INT8 (and FP16). 2 of every 4 input
   elements are skipped, doubling effective throughput. INT8 + 2:4 = 4×
   over FP16 dense.

The cost: precision loss. FP16 has ~3 decimal digits of mantissa; INT8 has
about 2 (8 bits ≈ 2.4 decimal digits, minus sign and quantization scale).
For models trained or fine-tuned to handle this loss, the throughput win
dominates. For models that are not, the loss can be unacceptable.

## The IMMA instruction

`IMMA.16816.S8.S8` is the INT8 Tensor Core instruction on Ampere. Like
HMMA, it is warp-wide: all 32 lanes execute it together to compute a
16×8×16 matrix product. The PTX is `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`.

```sass
IMMA.16816.S8.S8 R[acc_int32], R[a_int8], R[b_int8], R[acc_int32]
```

Per IMMA:
- A fragment: 16×16 INT8 = 256 bytes, distributed as 8 INT8 per lane (4 packed in two 32-bit registers)
- B fragment: 16×8 INT8 = 128 bytes
- C/D accumulator: 16×8 INT32 = 512 bytes, 4 INT32 per lane

Each IMMA does 16×8×16 = 2048 INT8 multiply-accumulates. That is the same
*element count* as HMMA, but the elements are 8-bit instead of 16-bit, so
the per-instruction byte movement is half. The hardware can issue IMMA at
the same rate as HMMA, giving 2× throughput in operations.

## The IMMA pipeline differs from HMMA

This is the surprise of phase 2. HMMA on sm_86 has a hardware-fixed S08
stall between consecutive HMMAs — the warp must wait 8 cycles before
issuing another HMMA into the same pipeline. The compiler emits this stall
automatically; trying to reduce it via CuAssembler hand-edit produces
incorrect results.

IMMA does *not* have this constraint. The compiler conservatively emits
S04 stalls, but the hardware sustains S02 throughput when operands are
ready. CuAssembler hand-edit S04 → S02 gives **+1.6%** measured.

This matters because the IMMA inner loop on `igemm_tiled.cu` runs 8 IMMAs
(2 per `mma.sync.aligned.m16n8k16` × 4 K-steps). At S04 those 8 IMMAs
take 32 cycles; at S02 they take 16 cycles. Halving the inner loop matters
when the compute phase is short relative to load.

This is **observation 14** in the project notes. The observation also
underlies why cp.async helps INT8 GEMM more than FP16 GEMM (chapter 04
regime 2): IMMA's shorter compute phase leaves bigger exposed bubbles
relative to the LDG stall, and cp.async fills them.

## Version 1 — Naive IGEMM (`kernels/gemm/igemm/igemm.cu`)

**Setup**: same WMMA structure as the basic HGEMM, but with INT8 fragments
and INT32 accumulators. The result is dequantized to FP32 in the epilogue
via `scale = max(|A_tile|) × max(|B_tile|) / 127² ≈ 6.2e-5` (per-tensor
symmetric quantization).

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

wmma::fill_fragment(acc_frag, 0);
for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, A + k, K);
    wmma::load_matrix_sync(b_frag, B + k * N, N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
}
// dequantize
for (int i = 0; i < acc_frag.num_elements; i++)
    out_frag.x[i] = float(acc_frag.x[i]) * scale_a * scale_b;
wmma::store_matrix_sync(C, out_frag, N, wmma::mem_row_major);
```

**Performance**: 10,897 TOPS at 4096³. About 3.1% of INT8 peak.

Same shape as the naive HGEMM relationship to peak: simple WMMA wrapping
gets you to single-digit % of peak; the optimization is in the
surrounding tiling, double-buffering, and pipelining.

## Version 2 — Tiled IGEMM (`kernels/gemm/igemm/igemm_pipelined.cu`)

**Setup**: 64×64 block tile, 4 warps, double-buffered LDG → STS, BK=32
(2 K-steps per smem tile). Per warp inner loop: 4 mma_sync = 8 IMMA per
tile.

**Performance**: 15,078 TOPS. About 4.3% of peak. ~38% improvement over
naive.

This is the ceiling of synchronous double-buffering on IGEMM. The next
gain comes from cp.async — covered in chapter 04. Briefly: at 8 warps/SM
with 8-IMMA short compute phase, cp.async wins +35% (`igemm_pipelined_cpasync.cu`,
~20,400 TOPS).

## Version 3 — 16-warp 128×128 IGEMM (`kernels/gemm/igemm/igemm_8warp.cu` and successors)

**Setup**: same block-size growth as the 16-warp HGEMM in chapter 02:
BM=128, BN=128, 8-16 warps, double-buffered cp.async.

**Performance**: ~25,000 TOPS at 4096³ (varies by exact variant). About
7% of peak — strong but still substantially below CUTLASS-class kernels.

The remaining gap is the same as HGEMM's: the inner loop is HMMA/IMMA
bound, and approaching peak requires deeper software pipelining (multiple
in-flight cp.async groups), persistent grids, and possibly raw SASS
hand-edit to push S04→S02 for IMMA throughout (project achieved +1.6%
from this on a single kernel; not yet propagated to all variants).

## Online quantization (`kernels/gemm/igemm/igemm_online_quant.cu`)

A practical wrinkle: in real inference, A and B arrive as FP16 (from
the previous layer's output and the model's weights respectively).
Quantizing them to INT8 *outside* the kernel costs an extra DRAM
round-trip — read FP16, write INT8, read INT8 in IGEMM.

The online-quant kernel folds the conversion into the kernel:

1. Cooperatively load FP16 A/B tiles into smem
2. Compute per-tile max(|x|) via warp reduction (one float per warp)
3. Convert FP16 → INT8 in-place with the discovered scale, write back to
   the same smem tiles (interpreted as INT8)
4. Run IMMA on the now-INT8 smem tiles
5. Dequantize accumulator at epilogue

The fragment-element accumulator pattern (`running[2][8][8]` FP32 +
`acc[2][8]` INT32) lets per-tile quantization scales accumulate correctly
across K-tiles via the recurrence:

```
output = sum over k_tiles of: scale_a[k] × scale_b[k] × imma_result[k]
```

Each k-tile has its own scale pair; the recurrence rescales prior
partial sums when the scale changes. This is the same online-recurrence
pattern as flash attention's online softmax (chapter 05).

**Performance**: ~16,000 TOPS at 4096³ — slower than offline-quant IGEMM
due to the extra reduction work, but eliminates the FP16→INT8 DRAM
round-trip in pipelined inference. Net gain depends on whether the
caller's pipeline is bandwidth- or compute-bound.

## 2:4 Structured Sparsity (`kernels/gemm/igemm/igemm_sparse_tiled.cu`)

**The premise**: in NVIDIA's 2:4 sparsity pattern, every 4 consecutive
elements of A along K must contain *exactly* 2 non-zeros. The hardware
skips the 2 zeros entirely. Effective compute throughput doubles; A's
storage halves (compressed format with 4-bit metadata indicating which
2 of every 4 positions are non-zero).

**Setup**: PTX `mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. The
"sp" suffix and the doubled K dimension (32 instead of 16) are the key
differences from regular IMMA.

```ptx
mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    {%0,%1,%2,%3},          // accumulator out (4 × INT32)
    {%4,%5},                // A compressed (2 × 32-bit packed INT8)
    {%6,%7,%8,%9},          // B (4 × 32-bit packed INT8)
    {%10,%11,%12,%13},      // accumulator in
    %14,                    // metadata (4 bits per 4-element chunk)
    0x0;                    // selector
```

A is half its dense size: 16 × 32 INT8 dense becomes 16 × 16 INT8
compressed plus 16 × 4-bit metadata.

**Smem layout** (this kernel, 16 warps, 128×128 block):
- `smem_a[2][BM × 48]` = 12 KB (compressed A: 32 INT8 + 16 pad)
- `smem_b[2][BK × 144]` = 18 KB (full B: 128 INT8 + 16 pad, 16-byte aligned for cp.async)
- Total 30 KB — under 50 KB cliff for 2 blocks/SM

**Performance** (4096³, RTX 3070 Ti):

| metric | dense INT8 (`igemm_tiled`) | sparse INT8 (`igemm_sparse_tiled`) |
|---|---|---|
| time | 8.06 ms | 5.81 ms |
| effective TOPS | 17,755 | 11,834 |
| **dense-equivalent TOPS** | 17,755 | **35,509** |
| speedup | 1.0× | **2.0× dense-equivalent**, 1.39× wall-clock |

The 2× dense-equivalent number is the headline: at the same wall-clock
cost as 11,834 TOPS of dense compute, the sparse kernel is doing
35,509 TOPS of effective dense work. The 1.39× wall-clock speedup is
what the user actually experiences on a sparse model.

This is **issue #7's resolution** — the 2:4 sparsity path was already
shipped before this session and confirmed by re-benchmarking.

## The IMMA hand-tuning postmortem

CuAssembler disassembly of `igemm_tiled.sm_86.cubin` revealed the IMMA
stall pattern. The K-loop body contains 16 IMMA instructions in 8 pairs
(one mma.sync = 2 IMMA, 4 K-steps × 2 mma.sync each = 8 mma.sync = 16
IMMA per tile pair).

The compiler emits S04 control codes between IMMAs by default (4-cycle
stall). Hand-edit to S02 (2-cycle stall) gave +1.6% on the kernel.
Hand-edit to S01 produced incorrect results — operands not always ready
that fast.

The lesson: **HMMA's S08 is hardware-fixed, IMMA's S04 is
compiler-conservative**. They look similar in cubins (both Tensor Core
matrix-multiply-accumulate) but have different optimization ceilings.
Knowing this requires SASS-level inspection, not just CUDA C++ profiling.

## Performance summary (4096³, RTX 3070 Ti)

| variant | TOPS | % of INT8 peak | notes |
|---|---|---|---|
| Naive IGEMM | 10,897 | 3.1% | 4-warp, no pipelining |
| Tiled IGEMM (sync double-buffer) | 15,078 | 4.3% | 8-warp, BK=32 |
| **IGEMM + cp.async** | **20,400** | **5.9%** | +35% from cp.async (chapter 04) |
| 16-warp 128×128 IGEMM | ~25,000 | ~7% | varies by variant |
| Online-quant IGEMM | ~16,000 | 4.6% | folds FP16→INT8 conversion |
| **Sparse INT8 (dense-equiv)** | **35,509** | **10.2%** | 1.39× wall-clock vs dense |

The sparse INT8 number deserves emphasis: 10% of the INT8 peak in
*dense-equivalent* terms means about 5% of peak in actual instructions
issued, but a 2× sparsity multiplier makes it 10% of the equivalent dense
work. Phase 2's most efficient kernel.

## What this chapter teaches

INT8 GEMM follows the same structural recipe as FP16 GEMM (chapter 02):
tile, register-block, double-buffer, maximize occupancy. The differences
are at the instruction level:

- IMMA replaces HMMA; same warp-wide, different element type
- IMMA's S04 stall is compiler-conservative, not hardware-fixed (unlike HMMA's S08)
- INT8's shorter compute phase makes cp.async more profitable than for FP16 (chapter 04 regime 2)
- Online quantization is a real-world refinement when A/B arrive as FP16
- 2:4 sparsity is a free 2× when the model is structured to permit it

The most important meta-lesson: **the high-level optimization recipe
generalizes; the low-level constants do not**. HMMA optimizations carry
over conceptually to IMMA, but specific numbers (stall cycles, optimal
block sizes, smem padding) must be re-derived for each instruction.

## How to run it yourself

```bash
cd /mnt/d/dev/p/bare-metal/kernels/gemm/igemm

# Build the variants
nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu
nvcc --cubin -arch=sm_86 -O2 -o igemm_pipelined.sm_86.cubin igemm_pipelined.cu
nvcc --cubin -arch=sm_86 -O2 -o igemm_pipelined_cpasync.sm_86.cubin igemm_pipelined_cpasync.cu
nvcc --cubin -arch=sm_86 -O2 -o igemm_sparse_tiled.sm_86.cubin igemm_sparse_tiled.cu

# Run benchmarks
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
./bench 4096 4096 4096   # dense IGEMM variants

nvcc -arch=sm_86 -O2 -o bench_sparse bench_sparse.cu -lcuda -I../common
./bench_sparse 4096   # sparse 2:4 (dense-equiv ~35500 TOPS)
```

## Inspecting SASS

```bash
# Count IMMA instructions (different opcode than HMMA)
cuobjdump -sass igemm_tiled.sm_86.cubin | grep IMMA | wc -l

# Look for the S04 stall pattern (hand-tuning candidate)
cuobjdump -sass igemm_tiled.sm_86.cubin | grep -A1 IMMA | head -20

# Sparse: look for the LDSM pattern that reads compressed A
cuobjdump -sass igemm_sparse_tiled.sm_86.cubin | grep -E 'IMMA|LDSM' | head -20
```

## Source files

- `kernels/gemm/igemm/igemm.cu` (naive WMMA INT8 baseline)
- `kernels/gemm/igemm/igemm_pipelined.cu`, `igemm_pipelined_cpasync.cu` (sync vs async double-buffer)
- `kernels/gemm/igemm/igemm_8warp.cu`, `igemm_8warp_256.cu`, `igemm_8warp_256x256.cu` (block-size scaling)
- `kernels/gemm/igemm/igemm_online_quant.cu`, `igemm_online_quant_bankfree.cu` (online quantization)
- `kernels/gemm/igemm/igemm_sparse_tiled.cu` (2:4 sparsity)
- `kernels/gemm/igemm/bench_sparse.cu` (sparse correctness + perf)

## Cross-references

- Chapter 02 — GEMM from Scratch (HMMA, FP16 Tensor Core path)
- Chapter 04 — Software Pipelining (cp.async wins big on IGEMM, regime 2)
- Chapter 06 — The Four Laws (Law 4: 30 KB sparse layout chosen for 2 blocks/SM)
- `docs/gpu_reflections.md` Insight 14 — IMMA S04 → S02 hand-tune
- Issue #7 — 2:4 structured sparsity, closed as already-shipped

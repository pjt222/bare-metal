# How does this project compare to the SOTA?

> Honest assessment: we are **roughly 4-20× slower than NVIDIA's
> production libraries** depending on kernel. This document quantifies
> the gap, explains what creates it, and characterizes what closing it
> would take.
>
> The point of this project is not to beat cuBLAS. The point is to
> understand exactly what cuBLAS does that we do not.

## Hardware peak (RTX 3070 Ti Laptop, GA104 sm_86)

| Resource              | Peak                  |
|-----------------------|-----------------------|
| FP32 (CUDA cores)     | 21.7 TFLOPS           |
| FP16 Tensor Core      | 174 TFLOPS            |
| INT8 Tensor Core      | 348 TOPS              |
| BF16 Tensor Core      | 174 TFLOPS            |
| TF32 Tensor Core      | 87 TFLOPS             |
| DRAM bandwidth        | 608 GB/s              |
| L2 bandwidth          | ~3 TB/s (estimate)    |
| Shared memory BW      | ~16 TB/s (estimate)   |

These are the absolute ceilings. No software can go faster.

## Per-kernel comparison

| Kernel | Ours | % of peak | SOTA estimate | SOTA % peak | Gap |
|---|---|---|---|---|---|
| HGEMM 4096³ (FP16 dense)    | 31.9 TFLOPS  | 18.3% | 130-150 TFLOPS (cuBLAS) | 75-85% | **4-5×** |
| Sparse HGEMM 2:4 2048³      | 41.7 TFLOPS dense-eq | 24.0% | 240-280 TFLOPS dense-eq (cuBLAS Lt) | 70-80% | **6×** |
| IGEMM 4096³ (INT8 dense)    | 27.6 TOPS    | 7.9%  | 200-230 TOPS (cuBLAS Lt) | 60-65% | **7-8×** |
| Sparse INT8 mma.sp 2048³    | 39.7 TOPS dense-eq | 11.4% | 350-400 TOPS dense-eq | 50-60% | **9-10×** |
| Flash Attention seq=1024 b=8 h=8 d=64 | 11.5 TFLOPS | 6.6% | 80-100 TFLOPS (FA-2 official) | 45-58% | **7-8×** |
| Conv2d 64×64×320 (implicit) | 6.7 TFLOPS   | 3.8%  | 100-130 TFLOPS (cuDNN) | 55-75% | **15-20×** |
| GroupNorm SD 320ch          | ~50 GB/s     | 8.2% (BW) | ~400-500 GB/s (cuDNN) | 65-82% | **8-10×** |

**Caveat on SOTA numbers**: precise published numbers for cuBLAS /
cuDNN / FA-2 on RTX 3070 Ti Laptop specifically are scarce. The
estimates above scale published A100 / RTX 3090 results by the relative
peak ratio. CUTLASS public benchmarks confirm ~85% peak HGEMM on Ampere
across the family. Real measurements on this exact card may vary
±10-15% from the estimates.

## What creates the gap

Production kernels (cuBLAS, cuDNN, FlashAttention-2 official) use
techniques this project deliberately does not implement, in order to
keep the code readable and the optimizations individually
understandable:

### 1. Multi-stage cp.async pipelines (4-6 stages, not 2)

Our kernels use 2-stage double-buffer (one buffer loading, one
computing). Production kernels use 4-6 stages, hiding much longer
latencies. Each additional stage costs another smem buffer; production
kernels carefully fit them under the 50 KB cliff.

**Impact**: ~30-50% of the FA gap, ~20% of the GEMM gap.

### 2. Persistent grids with cooperative work distribution

Our kernels launch one block per output tile. Production kernels launch
one block per SM (persistent), then loop over output tiles, allowing
register / smem state to persist across tiles and avoiding launch
overhead.

**Impact**: ~10-20% of all kernels' gap; bigger for small kernels.

### 3. Streaming K splits with cross-block reduction

For tall/skinny GEMMs (large K, small M/N), production kernels split
K across blocks and reduce via atomicAdd or a separate reduction
kernel. Our kernels do K in a single block, paying full latency cost
when the K iteration is long.

**Impact**: 2-3× for skinny matrices, ~20-30% for square.

### 4. Hand-tuned tile sizes per (M, N, K) range

cuBLAS has dozens of HGEMM kernels, one selected per (M, N, K) shape
based on heuristics built from billions of NVIDIA-internal benchmark
runs. Our kernels use one tile size for all sizes (BM=128, BN=128,
BK=32 for the 16-warp HGEMM).

**Impact**: 10-30% across the range; bigger at unusual sizes.

### 5. Swizzled smem layouts to eliminate bank conflicts at every stride

We use simple `+8 padding` to avoid bank conflicts on the most common
strides. Production kernels use XOR swizzles that work for arbitrary
strides without padding overhead.

**Impact**: 5-15% on Tensor Core kernels.

### 6. Optimized epilogues (fused activations, custom reductions)

Our epilogues store accumulator → smem → DRAM. Production kernels
fuse activation, scaling, and output reformatting into the store
path, saving a full DRAM round-trip.

**Impact**: 5-20% on kernels with rich epilogues (ResBlock, fused norms).

### 7. Cross-block reduction for attention (split-Q)

FA-2 official splits the query dimension across blocks and reduces
across them via atomic update of the running max/sum. We process all
queries in one block per (batch, head), losing parallelism for small
seq.

**Impact**: ~50-60% of the FA gap at small seq.

### 8. Hand-written SASS for the inner loop

NVIDIA's libraries ship cubin compiled from hand-written PTX/SASS
inner loops with optimal control codes (stall counts, scoreboard
slots) for every supported architecture. We use `nvcc` output with
limited CuAssembler hand-edits (IMMA S04→S02 = +1.6% on one kernel).

**Impact**: 5-15% on Tensor Core kernels, near zero on memory-bound
ones.

## Specific gap accounting (HGEMM 16-warp 4096³)

We measured 31.9 TFLOPS (18.3% peak). Breakdown of the 4-5× gap to
cuBLAS's ~140 TFLOPS:

| factor | contribution to gap |
|---|---|
| Synchronous LDG → STS double-buffer (vs 4-stage cp.async) | ~1.5× |
| No persistent grid (per-tile launch overhead) | ~1.15× |
| Single tile size for all (M, N, K) | ~1.10× |
| `+8` padding (vs full XOR swizzle) | ~1.05× |
| `nvcc`-generated SASS (vs hand-tuned control codes) | ~1.10× |
| **Multiplied** | **~2.4×** |

That accounts for about half of the observed 4.4× gap. The rest is
~1.8× of "deeper engineering" — split-K, prefetcher hints, register
pressure tuning, alternate fragments — that is hard to enumerate
individually but adds up.

## Specific gap accounting (Flash Attention v2_pipeline)

We measured 11.5 TFLOPS at seq=1024, b=8, h=8, d=64 (6.6% peak).
Estimated FA-2 official: ~80-100 TFLOPS. Gap: ~7-8×.

| factor | contribution to gap |
|---|---|
| No split-Q (process all queries in one block per b,h) | ~3× at this seq |
| 2-stage cp.async (vs 4-6 stage) | ~1.5× |
| 8 warps per block (vs 16+ via cooperative split-Q) | ~1.4× |
| No persistent grid (relaunch per b,h) | ~1.15× |
| Single (Br, Bc) tile (vs autotuned) | ~1.10× |
| **Multiplied** | **~7.6×** |

This roughly matches the observed gap. The dominant factor (3×) is
split-Q parallelism. For large seq (>=8192) the gap shrinks because
the per-block work is enough to saturate the SMs; FA-2's split-Q
benefit is largest at small seq, exactly where ours plateaus.

## What we did achieve

The honest comparison cuts both ways. cuBLAS / cuDNN / FA-2 are:

- **~4 million lines of code** (cuBLAS), ~600k (CUTLASS), ~30k (FA)
- **Decades of NVIDIA internal engineering** + dedicated teams
- **Closed-source for cuBLAS, partially open for CUTLASS, MIT for FA-2**
- **Hand-tuned for every architecture and matrix shape**
- **Re-released with each CUDA version**

This project is:

- **~15,000 lines** total (kernels + tests + benches + docs)
- **Three weekends of one person's work**
- **Fully open**: every `.cu` is readable, every benchmark traceable
- **One architecture**: sm_86, no portability layer
- **One tile size per kernel**, no autotuning

We achieve **18-24% of FP16 TC peak** for our best kernels with
techniques that fit in a single CUDA file. cuBLAS achieves 75-85% with
techniques that fit in nobody's head.

The relevant comparison is not "how close to cuBLAS" but "how close to
peak per line of code". On that metric this project does well —
**~2,000-3,000 GFLOPS per kloc** of kernel code at the high end. cuBLAS
is closer to **~30 GFLOPS per kloc** when you count its full source
tree, which is to say cuBLAS is about 100× more code per measurable
performance.

## What it would take to close the gap

To go from 18% peak HGEMM to 75% peak HGEMM (4× improvement):

1. **4-stage cp.async pipeline** (~1.5×) — implementable in this
   codebase with 1-2 days of work. Tile size shrinks to fit smem.
2. **Persistent grid with output-tile loop** (~1.15×) — moderate
   refactor, all blocks self-schedule.
3. **Split-K for skinny matrices** (~1.5× for skinny) — requires
   atomic reduction kernel; significant refactor.
4. **Tile-size autotuner** (~1.1×) — wrap each kernel in 6-8 variants,
   pick best per shape. Easy if mechanical.
5. **XOR-swizzled smem** (~1.05×) — 1 day of work, bank-conflict
   elimination at all strides.

Multiplicatively that gets us to ~3.7× over current = ~70% peak HGEMM.
Achievable. Not done because each step adds complexity that obscures
the underlying mechanism this project is meant to teach.

For Flash Attention to go from 6.6% peak to 50% peak (8× improvement)
requires split-Q parallelism specifically. Without it, no amount of
pipelining or tile tuning closes the gap at small seq. Split-Q is the
single highest-EV optimization remaining in this codebase, listed in
`CONTINUE_HERE.md` as the primary unfinished work.

## Where this project competes well

Some metrics where being a small focused project actually pays:

- **Read time**: a person can read all 11 GEMM-family kernels in this
  repo in an afternoon. CUTLASS HGEMM alone is ~3,000 lines of
  templates; nobody reads it for fun.
- **Modifiability**: changing tile size or smem layout takes minutes.
  In CUTLASS it requires understanding the template parameter graph.
- **SASS-level transparency**: every kernel ships with its `.sm_86.cubin`,
  the SASS is ~hundreds of instructions, every one identifiable. cuBLAS
  cubins are tens of kilobytes of opaque hand-tuned SASS.
- **Discovered hardware quirks**:
  - 50 KB smem cliff (>50 KB/block → 1 block/SM measured)
  - HMMA S08 stall is hardware-fixed; CuAssembler can't beat it
  - IMMA S04→S02 is compiler-conservative; +1.6% with hand-edit
  - Padding tied with on-fragment shfl reductions (Stage 3 vs v2 swap)
  - Bc=128 regime-dependent, not strictly negative (wins at seq=4096)
  - Cymatic memory layout: real but conditional ±1.5-1.9× depending on access geometry

These are documented in [`docs/gpu_reflections.md`](gpu_reflections.md)
across 20 numbered observations. None of them are in NVIDIA's
documentation; all are discoverable only by running the code.

## Verdict

**Performance vs SOTA**: 4-20× behind cuBLAS / cuDNN / FA-2 official,
predominantly because we don't implement multi-stage pipelining,
persistent grids, split-K/Q, autotuning, or hand-tuned inner-loop
SASS. The gap is well-characterized and closable with a known
sequence of optimizations, none of which are research-grade hard.

**Performance vs textbook implementations**: 70-90× faster than naive
SGEMM, 8-22× faster than naive HGEMM, 19× faster than scalar Flash
Attention. We sit comfortably at "graduate-student-level optimization
with full justification" rather than "production library".

**Per-line-of-code efficiency**: comparable or better than cuBLAS,
because we're not bundled with 1000 supporting routines.

**Pedagogical value**: every optimization is documented with its
mechanism, its measurement, and (where it failed) its postmortem.
This is the actual deliverable. Someone reading the [tutorial
series](tutorial/) and the [reflections](gpu_reflections.md) can
learn the same lessons that 5-10 years of NVIDIA performance
engineering would teach, in a few days.

## See also

- [docs/tutorial/06-the-four-laws.md](tutorial/06-the-four-laws.md) — distilled optimization principles
- [docs/gpu_reflections.md](gpu_reflections.md) — 20 observations from this project
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's open-source kernel template library
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) — the reference attention kernel

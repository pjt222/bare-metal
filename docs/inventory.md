# Kernel inventory

The canonical kernel index for this repository, grouped by content
and internal structure. Every directory under `kernels/<family>/` is
a regular kernel; there are no speculative or second-class entries.

## Headline performance

<sub>**Toolchain provenance.** All benchmark tables in this file were
measured on RTX 3070 Ti (GA104, sm_86, 46-SM laptop bin), CUDA 13.2 /
nvcc V13.2.78, driver 595.97.</sub>

![Kernel performance overview](figures/performance_overview.png)

Measured GFLOPS across all completed kernels, grouped by precision
class. Sparse 2:4 numbers are dense-equivalent (the multiply count the
sparse pattern would do as dense work). Each bar is annotated with
% of its precision-class peak (FP32 = 21.7 TFLOPS, FP16 TC = 174,
INT8 TC = 348; see [`../AGENTS.md`](../AGENTS.md) hardware constants).

### Top kernels (RTX 3070 Ti Laptop)

| Kernel                  | Size    | Time      | GFLOPS              | % peak  |
|-------------------------|---------|-----------|---------------------|---------|
| **Sparse HGEMM 2:4**    | 4096³   | —         | **41,721** (eq)     | 24.0%   |
| **Sparse INT8 mma.sp**  | 2048³   | —         | **39,674** (eq)     | 11.4%   |
| **HGEMM 16-warp**       | 4096³   | —         | **31,910**          | 18.3%   |
| **IGEMM 128×256**       | 4096³   | —         | **27,591**          | 7.9%    |
| **Flash Attention v2**  | seq=1024 b=8 h=8 | **1.53 ms** | **11,453** | 6.6% |
| **Conv2d implicit GEMM**| 64×64 320ch | **1.13 ms** | **6,687**       | 3.8%    |
| **Online FP16→INT8**    | 4096³   | —         | **17,070**          | 9.6%    |

> **Sparse HGEMM 2:4 — measured & reconciled 2026-06-03 ([#140](https://github.com/pjt222/bare-metal/issues/140) / [#143](https://github.com/pjt222/bare-metal/issues/143)).**
> The old "31.9 TFLOPS / dense-parity" wording (section A below / the kernel
> README) was a category error: 31.9 TFLOPS is the *dense* HGEMM baseline, not
> the sparse result. Under a host-side clock lock (`nvidia-smi.exe -lgc 1605`,
> the regime free of the 150 W power-cap bimodal), matched 1605 MHz 4096³ —
> sparse dense-eq **42,257** vs dense **31,886** GFLOPS = **1.33× over dense**
> (= 133%, confirming the long-standing 131% in `gpu_reflections.md`; the locked
> dense 31,886 ≈ the 31,910 literal). Between the 1× floor and 2× ceiling of 2:4
> sparsity; the 41,721 headline (4096³) is confirmed within spread. The old
> "4096³ 0.81× regression" is **refuted** — at matched 1605 MHz sparse 4096³ is
> at parity-or-above 2048³ (42,257 vs 40,980), not a 19% drop; the apparent
> regression was native-boost power-cap noise.

> **`igemm_sparse_tiled` 4096³ — documented reference, not a regression-gated baseline.**
> Measured 2026-05-22: **50,497 dense-equiv GFLOPS / 2.722 ms** at a
> host-side clock lock of 1605 MHz (median of 11 clean samples,
> spread 1.02×). This kernel is power-bound on the 150 W laptop —
> its bench averages 50 kernel launches and `SwPowerCap` throttles a
> varying fraction of them mid-run, so at native boost the averaged
> number is ~1.9× bimodal and there is no fair no-throttle baseline.
> Locking the SM clock (`nvidia-smi.exe -lgc 1605,1605`, Windows host,
> elevated) keeps every launch under the power budget and the number
> is stable. Above ~1605 MHz the kernel is on a power-bound plateau:
> 1710 MHz and 1785 MHz deliver the same ~50-53k throughput while the
> clock sags back below the lock. The old 2026-05-10 baseline (30,889)
> and the 2026-05-21 gated re-baseline (27,170) were both
> throttle-contaminated — the real figure is ~63% higher. Because
> `bench_regress.R` re-measures at native boost, this entry is removed
> from `data/baselines.json`; gating it needs a lock-aware harness.
> Sweep data: `scripts/probe/clock_lock_sweep.R` /
> `scripts/probe/clock_lock_sweep.rds`.

### Phase progression — naive SGEMM to sparse INT8

![Phase progression](figures/phase_progression.png)

Each layer of amortization compounds:

| step                       | mechanism                              | speedup vs naive |
|----------------------------|----------------------------------------|------------------|
| Naive SGEMM (Phase 1)      | one thread per output, FFMA in a loop  | 1.0×             |
| Tiled SGEMM (Phase 2)      | block tile, smem buffer                | 2.2×             |
| Register-blocked SGEMM     | each thread computes 8×8 outputs       | 10.9×            |
| HGEMM (basic WMMA)         | switch to FP16 Tensor Cores            | 17.0×            |
| HGEMM 16-warp 128×128      | 2 blocks/SM, double-buffered LDG       | **69.2×**        |
| Sparse HGEMM 2:4           | mma.sp, 50% structured zeros           | **90.5×**        |

Three orders of magnitude from a textbook GEMM. Each step is a single
optimization with a clear mechanism — no autotuning, no library magic.

### Flash Attention — 1.60× cumulative through three refactors

![FA optimization waterfall](figures/fa_waterfall.png)

The Flash Attention path peaked at ~7,154 GFLOPS for the original
register-PV kernel (`flash_attn_br16_regpv.cu`). Three structural
refactors brought it to **11,453 GFLOPS** = **1.60× cumulative**,
plateauing at ~6.6% of FP16 Tensor Core peak:

| step                        | technique                              | gain      |
|-----------------------------|----------------------------------------|-----------|
| `regpv` (baseline)          | register PV accumulation               | —         |
| lean state                  | smaller per-warp state, fewer LDS      | +6%       |
| Q reg cache                 | hold Q fragment in registers across K  | +16%      |
| `v2` (smem_work eliminated) | replace smem reduce with on-frag shfl  | +14%      |
| `v2_pipeline`               | cp.async double-buffer at 8 warps/SM   | +14%      |

Two failed experiments are kept as counter-examples: the original
synchronous pipeline at 4 warps/SM (cp.async loses), and Bc=128
tile size (loses at seq < 4096, wins +1.6% at seq = 4096 only).
Detailed walkthrough in
[`tutorial/05-flash-attention.md`](tutorial/05-flash-attention.md).

The classification is non-exclusive: a kernel that uses both HMMA
and SHFL.BFLY appears in both section A and section B. Columns are:
filesystem path, primary identity, peak measured number on
RTX 3070 Ti (sm_86), and the dominant SASS instruction family.

## A. General matrix multiply (GEMM family)

Tensor-Core or FFMA dominated. Inner loop is multiply-accumulate
into a register tile.

| Path | Variant | Peak | Lead SASS |
|---|---|---:|---|
| `kernels/gemm/sgemm/`                 | naive / tiled / register-blocked FP32 | ~1 TFLOPS at 2048³  | `FFMA` |
| `kernels/gemm/hgemm/`                 | tiled FP16 WMMA → 16-warp persistent  | **31.9 TFLOPS** at 2048³ | `HMMA.16816.F32` |
| `kernels/gemm/hgemm_sparse/`          | 2:4 structured sparse FP16            | 41,721 dense-eq GFLOPS at 4096³ — 1.33× dense, clock-locked 1605 ([#143](https://github.com/pjt222/bare-metal/issues/143)) | `HMMA.16816.SP` |
| `kernels/gemm/igemm/`                 | INT8 IMMA + cp.async pipelining       | 27.6 TOPS (8warp_256) at 2048³ | `IMMA.16816.S8.S8` |
| `kernels/convolution/conv2d/conv2d_implicit_gemm.cu` | reshapes conv into GEMM with on-the-fly index gen | 7.2 GFLOPS at SD 64×64×320 | `HMMA.16816.F32` |

Postmortems: [Obs N (sparse), HH (CUDA-13.2 IMMA), GG (implicit GEMM v2), II/JJ (cymatic on K/V).](gpu_reflections.md)

## B. Reductions & normalization

Two-pass kernels where the first pass is a tree-reduction across a
warp or block, the second is a per-element transform using the
reduced statistic. Showcase for `SHFL.BFLY` butterflies and the
`MUFU` special-function units.

| Path | Reduction scope | Lead SASS |
|---|---|---|
| `kernels/reductions/softmax/`               | warp (then row across SMs)         | `SHFL.BFLY` + `MUFU.EX2` |
| `kernels/reductions/layernorm/`             | block                              | `SHFL.BFLY` + `MUFU.RSQ` |
| `kernels/reductions/groupnorm/`             | group of channels                  | `SHFL.BFLY` + `MUFU.RSQ` + `MUFU.RCP` |

See [`fragment_shfl_reductions.md`](fragment_shfl_reductions.md) for
the butterfly-tree pattern catalog.

## C. Attention (fused softmax(QKᵀ/√d)V)

Online algorithm: streams Kᵀ + V tiles, maintains running row max
+ denom, never materializes the N² intermediate. All 21 variants
in `kernels/attention/flash_attention/` and the cross-attention kin in
`kernels/attention/cross_attention/`.

| Path | Tile / variant | Peak | Lead SASS |
|---|---|---:|---|
| `kernels/attention/flash_attention/flash_attn.cu`              | scalar reference          | (~0.4 TFLOPS)        | scalar FFMA |
| `kernels/attention/flash_attention/flash_attn_wmma.cu`         | WMMA fragments            | (~3 TFLOPS)          | `HMMA` |
| `kernels/attention/flash_attention/flash_attn_br16_regpv.cu`   | Br=16, V in registers     | **7.2 TFLOPS**        | `HMMA` + `SHFL` + `MUFU.EX2` |
| `kernels/attention/flash_attention/flash_attn_persistent.cu`   | persistent grid           | similar               | + persistent CTA |
| `kernels/attention/flash_attention/flash_attn_split_q.cu`      | + split-Q (frontier)      | TBD                   | + workload split |
| `kernels/attention/cross_attention/cross_attn_pipelined.cu`    | image-Q + text-KV + cp.async | (working)         | `HMMA` + `LDGSTS` |

The 21 phase-3 variants arc 19× over the scalar reference. See
[Obs C, D, E, F, J, JJ](gpu_reflections.md).

## D. Convolution (specialized GEMM)

Same multiply-accumulate, different operand-layout strategy.

| Path | Strategy | DRAM reads / output | Peak |
|---|---|---|---:|
| `kernels/convolution/conv2d/conv2d.cu`                 | direct, scalar FFMA      | 9× (re-read input)             | ~0.4 TFLOPS |
| `kernels/convolution/conv2d/conv2d_im2col.cu`          | explicit col-buffer + WMMA | 1×, but col buf in L2/DRAM    | ~3 TFLOPS |
| `kernels/convolution/conv2d/conv2d_implicit_gemm.cu`   | implicit GEMM v1         | 1×, no col buffer              | ~5 TFLOPS |
| `kernels/convolution/conv2d/conv2d_implicit_gemm_v2.cu`| implicit GEMM v2 (Obs GG) | 1×, no col buffer             | **6.7 TFLOPS** at 320 channels |
| `kernels/convolution/resblock/resblock_fused.cu`       | conv+norm+conv fused     | 1× × 3 stages                  | block runtime depends on conv backend (Obs R: 7× swap) |

Tutorial: [`tutorial/04-software-pipelining.md`](tutorial/04-software-pipelining.md).

## E. Elementwise / pointwise

One read, transform, one write. Bandwidth-bound. Showcase for
`MUFU` and `--use_fast_math` substitutions.

| Path | Operation | Lead SASS |
|---|---|---|
| `kernels/elementwise/timestep_emb/`            | sinusoidal positional encoding         | `MUFU.SIN` + `MUFU.COS` + `MUFU.EX2` |
| `kernels/elementwise/activations/`             | ReLU / GELU / Swish                    | `MUFU.EX2` (fast-math), `IMNMX` |

## F. Multi-kernel layer composition

Whole-layer pipelines that chain several of the above primitives in
one launch sequence. Tests whether per-kernel optimizations stack
when on-chip resources are shared.

| Path | Composition | Note |
|---|---|---|
| `kernels/composition/attention_layer/`     | QKV-proj (HGEMM) → FA → out-proj (HGEMM) → residual | ~35-40% of layer runtime is the two HGEMMs |

## G. Memory layout studies

The SASS body is a generic gather-sum loop; the *interesting* code
is the input-permutation table. These kernels measure layout
geometry, not compute throughput.

| Path | Pattern | Headline |
|---|---|---|
| `kernels/memory_layout/cymatic/` | Chladni-mode (n,m) gather indices over 2D disc | **1.53×** at sector midlines, **0.53×** at sector boundaries (Obs T) |

The cymatic kernel is a regular phase-4 kernel — same build, same
correctness check, same SASS-hand-edit eligibility. It's only
excluded from `bench_regress` because its bench output is a
multi-column row-vs-cym table that doesn't fit the single-number
baselines schema.

## H. Toolchain experiments

Pipeline experiments above SASS — alternative front-ends, codegen
backends. Behavior is identical at the cubin level; the question is
only what the developer-facing language emits.

| Path | Front-end | Compared to | Result |
|---|---|---|---|
| `experiments/rust-experiments/vecadd_oxide`     | Rust + cuda-oxide | nvcc vecadd | 2× SASS bloat (Obs KK) |
| `experiments/rust-experiments/cymatic_oxide`    | Rust + cuda-oxide | nvcc gather_sum | 0.67× SASS but 0.65–0.80× runtime; nvcc unroll heuristic dominates (Obs LL) |

## I. Hello-world / tutorial primitives

Toy kernels chosen for tooling demonstration, not for performance.

| Path | What it teaches |
|---|---|
| `kernels/tutorial/vector_add.cu` | The whole SASS hand-edit workflow: nvcc → cubin → cuobjdump → cuasmR FADD→FMUL → re-run |

---

## Cross-axis: by lead SASS instruction

Same kernels indexed by the instruction that defines them.

| SASS family | Where it dominates |
|---|---|
| `HMMA.16816.F32`           | A (hgemm), C (flash_attn, cross_attn), D (conv2d) |
| `HMMA.16816.SP` (sparse)   | A (hgemm_sparse) |
| `IMMA.16816.S8.S8`         | A (igemm) |
| `FFMA`                     | A (sgemm), D (direct conv2d), C (scalar FA) |
| `SHFL.BFLY`                | B (softmax, layernorm, groupnorm), C (FA softmax row max + denom) |
| `MUFU.EX2`                 | B (softmax), C (FA softmax), E (activations, timestep) |
| `MUFU.SIN`/`COS`           | E (timestep_emb) |
| `MUFU.RSQ`                 | B (layernorm, groupnorm) |
| `LDGSTS` (cp.async)        | A (igemm pipelined), C (FA pipeline / persistent variants), D (cross_attn pipelined) |
| `LDG`-only                 | E (activations), G (cymatic) |
| `FADD` → `FMUL` (hand-edit) | I (vector_add tutorial) |

## Cross-axis: by primary bottleneck

| Bottleneck | Kernels |
|---|---|
| Tensor Core throughput   | A.hgemm, A.hgemm_sparse, A.igemm (all at 2048³) |
| DRAM bandwidth           | D.direct conv2d (9× re-read), G.cymatic rowmajor_full, E.timestep_emb |
| Smem bank conflicts      | A.hgemm 4096³ (Obs HH), A.igemm pipelined |
| Software pipeline depth  | C.flash_attention v2_pipeline_pad, D.conv2d_implicit_v2 |
| Index decode             | D.conv2d_implicit_gemm at small Cin/Cout |

## See also

- [`gpu_reflections.md`](gpu_reflections.md) — observation catalogue with per-kernel postmortems.
- [`tutorial/`](tutorial/) — six-chapter prose walkthrough.
- [`comparison_to_sota.md`](comparison_to_sota.md) — measured gap to cuBLAS / cuDNN / cuSPARSELt per family.
- [`roofline_measured.md`](roofline_measured.md), [`sass_histogram.md`](sass_histogram.md), [`register_audit.md`](register_audit.md) — per-kernel measurement tables.
- Underlying data lives in [`../data/`](../data/).
- [`index.md`](index.md) — full documentation map.

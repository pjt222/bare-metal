# Kernels grouped by content / internal structure

Companion to [`kernels.md`](kernels.md), which lists kernels by
**development phase** (chronological). This file groups the same
kernels by **what's inside them** — the dominant SASS family, the
algorithmic pattern, the on-chip resource that bottlenecks them.
Every directory under `phase{1..5}/` is a regular kernel; nothing
here is "speculative" or second-class (audit Tier 12 dropped that
distinction).

The classification is non-exclusive: a kernel that uses both HMMA
and SHFL.BFLY appears in both A and B. The first column gives the
filesystem path; the second the *primary* identity, the third the
peak number on RTX 3070 Ti (sm_86).

## A. General matrix multiply (GEMM family)

Tensor-Core or FFMA dominated. Inner loop is multiply-accumulate
into a register tile.

| Path | Variant | Peak | Lead SASS |
|---|---|---:|---|
| `kernels/gemm/sgemm/`                 | naive / tiled / register-blocked FP32 | ~1 TFLOPS at 2048³  | `FFMA` |
| `kernels/gemm/hgemm/`                 | tiled FP16 WMMA → 16-warp persistent  | **31.9 TFLOPS** at 2048³ | `HMMA.16816.F32` |
| `kernels/gemm/hgemm_sparse/`          | 2:4 structured sparse FP16            | 31.9 TFLOPS dense-equiv at 2048³ | `HMMA.16816.SP` |
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
| `phase5/attention_layer/`     | QKV-proj (HGEMM) → FA → out-proj (HGEMM) → residual | ~35-40% of layer runtime is the two HGEMMs |

## G. Memory layout studies

The SASS body is a generic gather-sum loop; the *interesting* code
is the input-permutation table. These kernels measure layout
geometry, not compute throughput.

| Path | Pattern | Headline |
|---|---|---|
| `phase4/cymatic/` | Chladni-mode (n,m) gather indices over 2D disc | **1.53×** at sector midlines, **0.53×** at sector boundaries (Obs T) |

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

- [`kernels.md`](kernels.md) — the same kernels indexed by phase, with auto-generated headline tables and source paths.
- [`gpu_reflections.md`](gpu_reflections.md) — the full Obs catalog (~50 entries) with per-kernel postmortems.
- [`tutorial/`](tutorial/) — 6-chapter prose walkthrough that follows the phase ordering.
- [`comparison_to_sota.md`](comparison_to_sota.md) — gap analysis vs cuBLAS / cuDNN / FA-2 per family.

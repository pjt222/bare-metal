# Kernel Inventory & Comparison

**Single entry point** for all per-kernel comparison data in this
project (kernels indexed by **development phase**). For the
same kernels indexed by **content / internal structure** (GEMM
family, reductions, attention, convolution, elementwise, ...), see
[`kernels_by_family.md`](kernels_by_family.md).

Detailed views live in their own files (each linked below); this
page exists so you don't have to hunt across `docs/` to find the
right slice.

121 kernels measured, all on **RTX 3070 Ti Laptop (GA104, sm_86)**.
Runs are best-of-3 warm; SASS counts from cubins; NCU counters from
`results/ncu/all.csv`.

---

## Pick the right view

| You want to know…                                  | Look at                                      |
|----------------------------------------------------|----------------------------------------------|
| Top wins by precision class                        | [§ Headline wins](#headline-wins) below      |
| How my kernel scored on instruction mix            | [`sass_histogram.md`](sass_histogram.md) — 121 kernels, HMMA/IMMA/FFMA/MUFU/IMAD/BRA + useful_pct |
| Whether my kernel spills registers                 | [`register_audit.md`](register_audit.md) — 121 kernels, regs/block, spill flag, theoretical occupancy |
| Where my kernel sits on the roofline               | [`roofline_measured.md`](roofline_measured.md) — 10 NCU-profiled kernels, OI_DRAM/OI_L2 vs achieved GFLOPS |
| How far from cuBLAS / cuDNN / FA-2                 | [`comparison_to_sota.md`](comparison_to_sota.md) — 7 representative kernels, 4–20× gap factors |
| Whether my kernel still meets its locked baseline  | [`baselines.json`](baselines.json) — 6 anchor kernels, gates `bench_regress.R` |
| Why one kernel beats another structurally          | [`gpu_reflections.md`](gpu_reflections.md) — 36 observations (A–KK) |
| Hardware peaks for "% peak" denominators           | [§ Hardware ceilings](#hardware-ceilings) below |

---

## Headline wins

| Rank | Kernel                       | Precision | Size              | GFLOPS / TOPS    | % peak |
|-----:|------------------------------|-----------|-------------------|-----------------:|-------:|
| 1    | Sparse HGEMM 2:4             | FP16 TC   | 2048³             | **41,721 (eq)**  | 24.0%  |
| 2    | Sparse INT8 mma.sp           | INT8 TC   | 2048³             | **39,674 (eq)**  | 11.4%  |
| 3    | HGEMM 16-warp                | FP16 TC   | 4096³             | **31,910**       | 18.3%  |
| 4    | IGEMM 128×256                | INT8 TC   | 4096³             | **27,591**       | 7.9%   |
| 5    | Online FP16→INT8 GEMM        | INT8 TC   | 4096³             | **17,070**       | 9.6%   |
| 6    | Flash Attention v2           | FP16 TC   | seq=1024 b=8 h=8  | **11,453**       | 6.6%   |
| 7    | Conv2d implicit GEMM         | FP16 TC   | 64×64 320ch       | **6,687**        | 3.8%   |

*(eq = sparse-equivalent, i.e., dense-multiply count the sparse pattern would do.)*

Source for each row: see [§ Source paths](#source-paths) below.

---

## Hardware ceilings

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

These are the absolute denominators behind every "% peak" number in
this project.

---

## Aggregate insights (across all 121 kernels)

Three observations in [`gpu_reflections.md`](gpu_reflections.md) cover
the cross-kernel statistics (numbers reported as of the original Obs
run at 103 kernels; the 18 added since — mostly test cubins and the
experiments oxide spike — do not change the medians):

| Observation | Finding                                                                 |
|-------------|-------------------------------------------------------------------------|
| **AA**      | Median `useful_pct` = 12.5%; FFMA-dense kernels lead useful_pct, lose TFLOPS |
| **BB**      | All NCU-profiled kernels are **compute-limited** at measured OI         |
| **CC**      | **Zero register spills** across 103 kernels (max regs/block = 65,280)  |

Single best kernel-design takeaway: **the 50 KB shared-memory cliff is
load-bearing**. ≤50 KB → 2 blocks/SM → 8 warps (good); >50 KB → 1 block/SM →
4 warps (occupancy collapse, 2× regression measured). See
[`memory_hierarchy.md`](memory_hierarchy.md) and Obs FF (#95).

---

## Phase progression — naive SGEMM to sparse INT8

Each layer of amortization compounds. Top progression demonstrates the
~17× chain from naive FP32 to FP16 Tensor Cores; sparse + epilogues
push to 41,721 GFLOPS sparse-equivalent.

| Step                       | Mechanism                              | Speedup vs naive |
|----------------------------|----------------------------------------|-----------------:|
| Naive SGEMM (Phase 1)      | one thread per output, FFMA in a loop  | 1.0×             |
| Tiled SGEMM (Phase 2)      | block tile, smem buffer                | 2.2×             |
| Register-blocked SGEMM     | each thread computes 8×8 outputs       | 10.9×            |
| HGEMM (basic WMMA)         | switch to FP16 Tensor Cores            | 17.0×            |
| HGEMM 16-warp 128×128      | full SM occupancy, async pipelined     | (31,910 GFLOPS)  |
| Sparse HGEMM 2:4           | mma.sp, half the multiplies            | (41,721 dense-eq)|

Figure: [`figures/phase_progression.png`](figures/phase_progression.png).

---

## Local reference gap (measured only)

This section no longer uses extrapolated SOTA estimates. It now reports
only **locally measured reference-library comparisons**. Full pipeline:
[`comparison_to_sota.md`](comparison_to_sota.md).

| Workload | Ours | Local reference | % of reference |
|---|---:|---:|---:|
| HGEMM 16-warp 2048³ | 31,875 GFLOPS | 28,631 GFLOPS (cuBLAS) | **111.3%** |
| HGEMM 16-warp 4096³ | 31,765 GFLOPS | 29,708 GFLOPS (cuBLAS) | **106.9%** |
| IGEMM pipelined cp.async 4096³ | 20.23 TOPS | 29.44 TOPS (cuBLAS) | **68.7%** |
| Sparse IGEMM tiled 2048³ | 31.59 TOPS | 124.28 TOPS (cuSPARSELt) | **25.4%** |
| Sparse IGEMM tiled 4096³ | 30.89 TOPS | 170.11 TOPS (cuSPARSELt) | **18.2%** |
| Conv2d implicit GEMM 1×64×64×320×320 | 7,150 GFLOPS | 16,910 GFLOPS (cuDNN) | **42.3%** |

Not measured locally yet:

- Flash Attention — installed cuDNN headers on this machine do not expose the graph-based SDPA frontend needed for a direct local reference
- GroupNorm — no direct local cuDNN GroupNorm harness yet

---

## NCU-measured roofline (10 tensor-active)

Operational intensity is **measured**, not estimated. All 10 sit
**above the L2 ceiling** → compute-bound (Obs BB). Full table with
DRAM/L2 bytes per launch: [`roofline_measured.md`](roofline_measured.md).
Figure: [`figures/roofline_measured.png`](figures/roofline_measured.png).

| Kernel                                | OI_DRAM | OI_L2 | Achieved GFLOPS |
|---------------------------------------|--------:|------:|----------------:|
| HGEMM 16-warp (4096³)                 | 162.3   | 41.8  | 29,751          |
| Sparse INT8 GEMM (4096³)              | 126.2   | 34.3  | 18,035          |
| HGEMM 16-warp+epi (4096³)             | 162.5   | 62.0  | 16,730          |
| HGEMM 256×128 (4096³)                 | 273.6   | 81.8  | 16,073          |
| FA v2 pipeline (seq=1024,b=8,h=8)     | 412.4   | 56.8  | 8,892           |
| FA v2 baseline                        | 412.9   | 57.8  | 7,773           |
| FA v2 persistent                      | 411.2   | 57.3  | 7,227           |
| Cross-attn v2 (1024 q, 256 kv, h=8)   | 168.1   | 44.6  | 6,417           |
| FA regpv (legacy)                     | 242.4   | 30.8  | 5,554           |
| ResBlock implicit GEMM (320ch)        | 419.5   | 19.0  | 2,092           |

---

## Regression baselines (6 anchors)

Locked best-of-3 timings. `scripts/bench/bench_regress.R` fails on >10%
drop. Source: [`baselines.json`](baselines.json).

| Kernel                              | Size               | ms     | GFLOPS / TOPS |
|-------------------------------------|--------------------|-------:|--------------:|
| HGEMM 16-warp                       | 2048³              | 0.527  | 31,910        |
| HGEMM 16-warp                       | 4096³              | 4.220  | 31,910        |
| Sparse INT8 tiled (post-#65)        | 2048³              | 0.433  | 39,674        |
| Sparse INT8 tiled (post-#65)        | 4096³              | 4.317  | 31,835        |
| IGEMM pipelined cp.async            | 4096³              | 6.600  | 20,688        |
| Flash Attention br16 regpv          | seq=1024 b=8 h=8   | 2.810  | 6,112         |
| Conv2d implicit GEMM                | 1×64×64×320×320    | 1.130  | 6,687         |

---

## SASS instruction mix (top 10 by useful_pct)

`useful_pct = (HMMA + IMMA + FFMA + FMUL + FADD) / total_inst`. Full
103-row table: [`sass_histogram.md`](sass_histogram.md). Figure:
[`figures/sass_histogram.png`](figures/sass_histogram.png).

| Kernel                            | Total | HMMA | FFMA | useful_pct |
|-----------------------------------|------:|-----:|-----:|-----------:|
| sgemm_register_blocked            | 1016  | 0    | 512  | 50.4%      |
| flash_attn_multihead              | 1224  | 0    | 138  | 40.8%      |
| flash_attn_br16_v2_bc128          | 1496  | 128  | 13   | 40.4%      |
| flash_attn_1warp                  | 1272  | 0    | 138  | 39.3%      |
| gelu_kernel                       | 72    | 0    | 4    | 38.9%      |
| flash_attn_4warp                  | 4152  | 0    | 264  | 35.8%      |
| flash_attn_br16_v2                | 1024  | 64   | 13   | 34.1%      |
| cross_attn_v2                     | 1040  | 64   | 13   | 33.6%      |
| flash_attn_v2_persistent          | 1088  | 64   | 13   | 32.1%      |
| flash_attn_bc128                  | 4080  | 128  | 70   | 31.2%      |

(Note: high useful_pct ≠ high TFLOPS. FFMA-dense `sgemm_register_blocked`
tops the list at 50.4% useful but delivers ~2 TFLOPS; HMMA-dense
HGEMM 16-warp sits at ~24% useful but delivers 31.9 TFLOPS. See Obs AA
for the trade-off discussion.)

---

## Source paths

Headline-7 kernels and where their source lives:

| Headline kernel              | Source path                                              |
|------------------------------|----------------------------------------------------------|
| Sparse HGEMM 2:4             | `kernels/gemm/hgemm_sparse/`                                   |
| Sparse INT8 mma.sp           | `kernels/gemm/igemm/igemm_sparse_tiled.cu`                     |
| HGEMM 16-warp                | `kernels/gemm/hgemm/hgemm_16warp.cu`                           |
| IGEMM 128×256                | `kernels/gemm/igemm/igemm_8warp_*.cu`                          |
| Online FP16→INT8 GEMM        | `kernels/gemm/igemm/igemm_online_quant*.cu`                    |
| Flash Attention v2           | `kernels/attention/flash_attention/flash_attn_br16_v2*.cu`          |
| Conv2d implicit GEMM         | `kernels/convolution/conv2d/conv2d_implicit_gemm.cu`                  |

Phase directories (kernel families):

| Phase   | Path                              | Family                                         |
|---------|-----------------------------------|------------------------------------------------|
| 1       | `kernels/tutorial/`                         | Vector add (SASS hello world, FADD→FMUL)       |
| 2       | `kernels/gemm/sgemm/`                   | Naive → tiled → register-blocked SGEMM         |
| 2       | `kernels/gemm/hgemm/`                   | WMMA → 16-warp 128×128 HGEMM                   |
| 2       | `kernels/gemm/hgemm_sparse/`            | 2:4 sparse mma.sp                              |
| 2       | `kernels/gemm/igemm/`                   | INT8 IMMA + sparse + online quant              |
| 2       | `kernels/reductions/{softmax,layernorm}/`, `kernels/elementwise/activations/` | Reductions + MUFU SFU                  |
| 3       | `kernels/attention/flash_attention/`         | Scalar → 4-warp → Br=16 HMMA → v2 → pipeline   |
| 4       | `kernels/convolution/conv2d/`                  | Direct 9× → implicit GEMM (22× win)            |
| 4       | `kernels/{reductions/groupnorm,convolution/resblock,attention/cross_attention,elementwise/timestep_emb}/` | UNet primitives        |
| 4       | `kernels/memory_layout/cymatic/`                 | Chladni-pattern memory layout study            |
| 5       | `kernels/composition/attention_layer/`         | Multi-head attention layer composition         |
| Exp     | `experiments/rust-experiments/`        | cuda-oxide Rust→PTX spike (Obs KK)             |
| Exp     | `experiments/rust-experiments/cymatic_oxide/` | cuda-oxide on gather_sum: SASS shorter, runtime slower (Obs LL) |

---

## Regenerating the data

All comparison tables are regenerable from cubins + NCU runs:

| Run                                      | Updates                                          |
|------------------------------------------|--------------------------------------------------|
| `Rscript scripts/audit/sass_histogram.R`       | `sass_histogram.{md,csv}` + figure (103 kernels) |
| `Rscript scripts/register_audit.R`       | `register_audit.{md,csv}` (103 kernels)          |
| `Rscript scripts/profile/roofline_measured.R`    | `roofline_measured.md` + figure (NCU-profiled)   |
| `Rscript scripts/bench/bench_regress.R`        | Pass/fail vs `baselines.json`                    |
| `Rscript scripts/bench/bench_flash_all.R`      | FA-variant comparison table                      |
| `Rscript scripts/audit/generate_readme_figures.R` | Top-level README figures                      |

This index page is hand-curated, not auto-generated. Update it when
adding a new kernel family, observation, or detail-doc.

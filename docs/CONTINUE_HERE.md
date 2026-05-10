# Continue Here

> **Last updated**: 2026-05-08 (NCU profiling session — 7 issues closed,
> +8 padding pattern established across 5 kernels, 1.4-2.4× wins each)
>
> **Session status**: 4 commits, 7 issues closed (#84, #85, #88, #89,
> #97, #98, #99). NCU bank-conflict counter unlocked diagnosis-and-fix
> pattern: measure conflict rate, add +8 row stride padding, gain
> 1.4-2.4×. Five of six measured kernels needed the fix.
>
> **New canonical kernels**:
> - `flash_attn_br16_v2_pipeline_pad2.cu` (FA, 21.4 TFLOPS at seq=1024,
>   2.36× over previous canonical)
> - `flash_attn_br16_v2_pad.cu` (FA baseline, 1.91×)
> - `flash_attn_v2_persistent_pad.cu` (FA persistent, 2.08×)
> - `cross_attn_v2_pad.cu` (cross-attn typical, 1.91×; regime-dependent)
> - `hgemm_16warp_epi_pad.cu` (HGEMM epi, 1.41×)
>
> **Open issue queue**: 11 issues. Most recent diagnostic findings
> (Observations U/V/W/X/Y) reshape what's worth pursuing next — see
> "Next sprint priorities" section below.
>
> **NCU-driven findings (Observations U-Y)**:
> - U: smem traffic, not HMMA S08, was FA bottleneck (refuted #84 split-Q)
> - V: HGEMM `math_pipe_throttle` is HMMA queue pressure, not address
>   arithmetic (refuted IMAD-chain hypothesis, 1.01× negative result)
> - W: FA pipeline 2.36× from +8 padding, bank conflict rate 87.5→ 2.2%
> - X: pattern generalizes to FA baseline / persistent / cross-attn
>   (1.91-2.08× each)
> - Y: HGEMM bank audit confirms hgemm_16warp clean (0.17% rate),
>   reveals hgemm_16warp_epi at 75.9% rate, fix gives 1.41×
>
> **NCU harness shipped**:
> - `scripts/profile/ncu_profile.R` (267 lines, single kernel + `--dry-run`)
> - `scripts/profile/ncu_profile_all.sh` (sweep across phase 2-4)
> - `docs/ncu_metrics.md` (full per-kernel diagnosis tables)
> - `results/ncu/all.csv` (raw output, 10 kernel configs × 15 metrics)
>
> **Three assumptions overturned (Observation U)**:
> 1. FA plateau is smem-traffic-bound, not HMMA S08-bound
>    (`stall_mio + stall_short_sb = 12.3` vs `stall_wait = 0.93`)
> 2. HGEMM 16-warp gap is FFMA pipe oversubscription
>    (`stall_math_throttle = 35.46`), not Tensor Core density
> 3. HGEMM 16-warp+epi epilogue is a regression
>    (`stall_mio=21`, `stall_barrier=17`, `stall_short_sb=17`)
>
> **NCU prerequisite**: GPU performance counters must be enabled on the
> Windows host (NVIDIA Control Panel → Developer Settings → Manage GPU
> Performance Counters → "Allow access to all users") and host rebooted.
> Once enabled, WSL-side `ncu` works without sudo.
>
> **Tutorial series (`docs/tutorial/`) all 6 chapters complete in full prose**:
> ~80 KB / ~20,000 words across 01 SASS, 02 GEMM, 03 INT8, 04 Pipelining,
> 05 Flash Attention, 06 Four Laws.
>
> **Cymatic memory layout** (`phase4/cymatic/`, `scripts/cymatic_*.R`,
> `docs/cymatic_memory_mapping.md`): Chladni-pattern memory layout for
> circular access geometry. R generator + CUDA gather bench measured at
> GRID=2048² (DRAM regime). Conditional speedup: **+1.53× at sector
> midlines, −1.89× at boundaries**. Documented in Observation T.
>
> **Warmup**: `Rscript scripts/verify_setup.R`

---

## Verified Headline Numbers (post-warmup, 3-run mean)

| target | before | after | speedup |
|---|---|---|---|
| Flash Attention seq=1024 b=8 h=8 | 7154 GFLOPS (regpv) | **11453 GFLOPS** (v2_pipeline) | **1.60×** |
| Flash Attention smem-only path | 7154 (regpv) | **9998** (v2 nosmem) | **1.40×** |
| Cross-attention typical (1024×256) | 4036 GFLOPS | **6550 GFLOPS** | **1.62×** |
| Cross-attention CLIP-77 (256×77) | 656 GFLOPS | 530 GFLOPS | **0.81×** (regime loss) |
| ResBlock SD UNet (N=1 C=320 H=W=32) | 13.07 ms (289 GFLOPS) | **1.86 ms (2025 GFLOPS)** | **7.01×** |
| Sparse INT8 IGEMM 4096³ | 15030 TOPS dense | **35509 dense-equiv TOPS** | **1.39×** (already shipped) |

Flash Attention plateau: ~11.5 TFLOPS = 6.6% of FP16 TC peak (174 TFLOPS).
Path: `regpv → lean state → Q reg cache → smem_work elimination (1.40×)
       → cp.async at 8 warps (additional +0.20×)`.

---

## Key Insights From This Session

### 1. Fragment-shfl reductions: a reusable pattern

When a Tensor Core kernel round-trips an accumulator through smem to compute
a per-row reduction (max, sum, dot), the round-trip can be eliminated by
reducing directly on fragment elements via intra-group `__shfl_xor_sync`
(offsets 1 and 2, covering all 4 lanes within each row group).

In WMMA m16n16k16 each lane "owns" 2 specific rows (`row_lo = groupID`,
`row_hi = groupID + 8`). The 4 lanes per row group hold all 16 cols of the
tile collectively. Combined with FP16 weights written direct-to-smem at
WMMA-row-major positions and `pv_accum` written direct-to-global at the end,
the 16 KB FP32 score buffer disappears entirely.

**Result on Flash Attention**: LDS+STS instructions in cubin 238 → 30,
smem 32 KB → 24 KB, throughput +40% across all sizes.

**Generalizes to**: any softmax-like, layer-norm-like, or group-norm-like
kernel that uses Tensor Core matmuls with online reductions.

**Documented**: `docs/fragment_shfl_reductions.md`

### 2. Padding-vs-occupancy tradeoff is empirical, not abstract

Standard "pad to break bank conflicts" guidance fails when the kernel is
already at high occupancy. At 12 warps/SM (3 blocks × 4 warps), 8-way
ldmatrix replays are hidden by warp scheduling. Adding +1 KB padding to
"fix" the conflicts pushes occupancy to 8 warps (2 blocks/SM), exposing
per-warp serialization more than padding eliminates.

Empirical result: padding lost 20-32% across all sizes on the
`flash_attn_br16_regpv` kernel.

**Predictor**: `pad_tradeoff(conflict_factor, ldsm_per_iter, hmma_per_iter,
warps_no_pad, warps_pad)` in `scripts/audit/ldmatrix_conflicts.R`. Calibrated
against measurement: predicts 0.86× when measured was 0.81×.

**Lesson**: when smem padding fails due to occupancy regression, consider
eliminating the smem allocation entirely (insight 1) rather than working
around the conflict.

### 3. Optimizations are occupancy-dependent — re-run the catalog when occupancy changes

Multiple optimizations flipped status when the kernel's smem footprint
changed:

| optimization | original kernel (8 warps) | v2 kernel (12 warps) | v2 pipeline (8 warps) |
|---|---|---|---|
| cp.async pipelining | -5% (lost) | n/a | **+14-41% (won)** |
| Persistent grid | +10% at large tiles | -8% to +12% (regime mix) | (not measured) |
| KV/W padding | -20-32% | tied (±3%) | (not measured) |

The same optimization can lose at 4 warps, win at 8 warps, lose again at 16
warps. When introducing any change that affects occupancy, re-run the
catalog of "established" wins — their status may have flipped.

### 4. Pattern has a regime boundary at ~4 outer iterations

The fragment-shfl pattern (insight 1) has a fixed kernel-level overhead:
Q register cache prologue, pv_accum fill, direct-to-global epilogue.
These execute once per kernel call, but per-iteration smem savings scale
with iteration count.

| outer iters | regime | example | result |
|---|---|---|---|
| ~16 (FA seq=1024) | clear win | flash_attn_br16_v2 | 1.40× |
| ~4 (cross-attn 256 KV) | clear win | cross_attn_v2 | 1.43× |
| ~2 (CLIP-77, 77 KV) | loss | cross_attn_v2 at small seq_q | 0.81× |

**Production guidance**: dispatch on workload size. Roughly,
`(outer_iters * inner_dim >= some_threshold) ? v2 : baseline`.

### 5. Read benchmark data row-by-row, not just the headline

My own Bc=128 benchmark this session showed loss at seq=512/1024/2048 and a
small *win* at seq=4096. I summarized it as "cleanly negative" without
reading the seq=4096 row carefully. Heal pass caught the error.

The pattern is real: optimizations that lose at small problem sizes can
flip to wins at large problem sizes when iteration count amortizes a fixed
overhead. Same pattern as cross-attention CLIP-77 split (loses) vs typical
1024×256 (wins).

**Discipline**: when reporting an optimization result across a sweep, list
the per-row delta. Do not collapse to a single "loses N%" number unless
every row in the sweep has the same sign. The fix is mechanical: include
the full table in any documentation, and read every row before writing
the conclusion.

### 6. Warmup discipline matters for bench claims

Cold-cache numbers can be 10-20% off from warm-cache steady state.
Headline cumulative speedups should be verified with all variants
measured in the same warmup regime — not by combining best-of-runs from
different sessions. The original "1.86× cumulative FA speedup" claim was
based on extrapolated time; verified post-warmup it's 1.60×. Same shape,
smaller magnitude.

**Discipline**: 5-iteration warmup, then 3-run mean. Same seq/batch/heads
across all variants in the comparison. Document the warmup regime when
publishing numbers.

### 7. ResBlock case: count DRAM passes before optimizing FLOPS

`kernels/convolution/resblock/bench.cu` was running `conv2d_nhwc` (direct FFMA, reads
input X 9× per output element). Achieved 289 GFLOPS at SD UNet config.
The codebase already had `implicit_gemm_conv` (Tensor Cores, im2col on the
fly, reads X exactly once) achieving 4800-6800 GFLOPS standalone.

**The fix was kernel selection, not kernel optimization.** Swapping
delivered 7.01× ResBlock speedup. No new code beyond a host-side weight
reshape (FP32 → FP16) and an updated launch config.

**Lesson**: before optimizing the inner loop, count DRAM passes per byte.
A kernel reading every byte 9 times is a 9× bandwidth problem disguised
as a FLOPS problem.

### 8. Empirical retest of "deprioritized" issues confirms deprioritization

#18 (128×256 tiles) and #80 (XOR swizzle) had been deprioritized via reasoning,
not measurement. This session retested both empirically:

- **#18 retest**: Build attempt (`igemm_online_quant_bf_128x256.cu`) hit 3 distinct constraints — 64 KB static smem (over 48 KB compile max), 211 regs/thread vs 128/thread budget at 16 warps × 1 block, and Observation S evidence on a parallel kernel that bigger tiles at lower occupancy lose. Three independent blockers, all measured.
- **#80 retest**: Cubin instruction analysis showed LDSM is 4.6% of total instructions on `flash_br16_v2.sm_86.cubin`. Stage 3 padding (alternative conflict-elimination mechanism) was tied with v2 within ±3%. Both pieces of evidence say XOR swizzle would address a non-bottleneck.
- **Bc=128 v2 retest**: a hopeful experiment (Next-Step suggestion A.1 in this file). Built `flash_attn_br16_v2_bc128.cu`, measured: lost 5.8-7.4% at seq ∈ {512, 1024, 2048}, **won 1.6% at seq=4096**. Loses 19-23% to v2_pipeline at same 2 blocks/SM occupancy at all sizes. **Regime-dependent**, not uniformly negative — Observation S. Crossover lies between seq=2048 and seq=4096.

**Lesson**: "deprioritized" issues are worth a quick empirical retest before
closing. The retest either confirms the deprioritization or surfaces a
regime-specific surprise. Bc=128 retest in particular found a small win at
seq=4096 that I initially summarized away as "loses everywhere" — honest
re-reading of my own benchmark output (during a heal pass) caught the error.
Lesson within the lesson: read the data row-by-row, not just the headline.

### 9. The four laws still hold (and reinforce each other)

After 18 observations across 5 phases, the four laws of GA104 are
confirmed by every successful and failed optimization in this session:

1. **Feed Tensor Cores continuously** — cp.async wins only when warp count is high enough to schedule it (insight 3 above)
2. **Read each byte of DRAM exactly once** — implicit GEMM saves 9× re-reads (insight 6)
3. **Fill the warp schedulers** — 12 warps hide bank conflicts that 8 warps cannot (insight 2)
4. **Never cross the 50 KB smem cliff** — smem_work elimination kept 24 KB at 3 blocks/SM (insight 1)

The laws are co-dependent. Most failures violate two or more simultaneously.
Most wins respect all four.

**Documented**: `docs/tutorial/06-the-four-laws.md` (the synthesis chapter)
and `docs/tutorial/05-flash-attention.md` (the deep case study, also complete
this session).

---

## Issues Closed This Session

| # | title | resolution |
|---|---|---|
| 4  | Fuse GroupNorm into Conv2d epilogue | Not planned — GN was 0.1% of ResBlock; real bottleneck was conv2d (resolved via #83) |
| 7  | 2:4 structured sparsity with IMMA | Already shipped in `kernels/gemm/igemm/igemm_sparse_tiled.cu`; documented and closed |
| 17 | smem padding for ldmatrix bank conflicts | Not planned — padding regresses occupancy (Observation O); use fragment-shfl pattern instead |
| 18 | 128×256 tiles for online-quant kernel | Not planned — 3 independent blockers (static smem max, reg budget, Observation S) |
| 29 | Apply tiled HGEMM techniques to FA | Three-stage refactor delivered 1.40× (Observation P) |
| 78 | Apply nosmem to FA pipeline + persistent | Pipeline +14-41% (Observation Q); persistent neutral-to-loss at 12 warps |
| 79 | Apply nosmem to cross_attention | Wins 1.43-1.62× at typical sizes; loses 19% at CLIP-77 (regime split) |
| 80 | XOR swizzle exploration on top of nosmem | Not planned — LDSM is 4.6% of cubin, hidden at 12 warps; addresses non-bottleneck |
| 81 | Promote nosmem variant to canonical | Renamed to `flash_attn_br16_v2.cu`; Phase 3d README section added |
| 82 | Document fragment-shfl reduction pattern | `docs/fragment_shfl_reductions.md` written and linked from README |
| 83 | Swap conv2d_nhwc → implicit_gemm_conv in ResBlock | `bench_implicit.cu` delivered 1.80-7.01× across configs |

## Open Issues

| # | title | priority | status |
|---|---|---|---|
| 32 | Research: polyhedral spring networks | low | Research placeholder, outside core kernel scope |

Only placeholder issue remains. Active CUDA optimization queue: empty.

## Reprioritization (post-NCU, 2026-05-08)

### Five steps executed this session (commits ef77e0d, 37bc94a, f2100ac)

1. **NCU diagnostics harness (#89)** — `scripts/profile/ncu_profile.R`,
   `ncu_profile_all.sh`, `docs/ncu_metrics.md`. 15 metrics, validated
   against `ncu --query-metrics --chip ga104`. Now in routine use.

2. **HGEMM IMAD-chain hypothesis (Obs V)** — falsified. Aligned variant
   cut IMAD 24% / ISETP 91% / BRA 60%, perf 1.01× flat. Reframed
   `math_pipe_throttle` as HMMA queue pressure.

3. **FA pipeline +8 padding (#88, Obs W)** — 2.36×. New canonical.

4. **+8 padding generalized (#97, Obs X)** — 4 more kernels: FA
   baseline, persistent, cross-attention, all 1.91-2.08×.

5. **HGEMM bank audit (#98, #99, Obs Y)** — hgemm_16warp clean
   (0.17%); hgemm_16warp_epi was 75.9%, fix gives 1.41×.

### Closed this session

| issue | resolution |
|---|---|
| #84 split-Q FA | not-planned for trained shapes (NCU showed no SM starvation) |
| #85 4-stage HGEMM pipeline | re-evaluate (TC util already 46%, gap is HMMA queue) |
| #88 XOR-swizzle | resolved by simpler +8 padding (Obs W) |
| #89 NCU harness | DONE |
| #97 +8 to other TC kernels | DONE (4 kernels, 1.9-2.1×) |
| #98 HGEMM bank audit | confirms existing padding clean |
| #99 hgemm_epi over-syncing | fixed via +8 padding (1.41×) |

### Headline numbers updated

| kernel | before | after | speedup | % FP16 TC peak |
|---|---:|---:|---:|---:|
| FA pipeline seq=1024 b=8 h=8   | 11.5 TFLOPS | **21.4** | **2.36×** | 12.3% |
| FA pipeline seq=512             | 8.4         | 20.6     | 2.44×    | 11.8% |
| FA pipeline seq=4096            | 10.1        | 21.1     | 2.08×    | 12.1% |
| FA baseline seq=1024            | 7.9         | 15.1     | 1.91×    | 8.7%  |
| FA persistent seq=1024          | 7.3         | 15.2     | 2.08×    | 8.7%  |
| Cross-attn typical (1024×256)   | 5.5         | 10.5     | 1.91×    | 6.0%  |
| HGEMM 16-warp 4096³             | 30.0 (clean baseline) | n/a | n/a | 17.2% |
| HGEMM 16-warp_epi 4096³          | 18.0        | 25.4     | 1.41×    | 14.6% |

### Next sprint priorities (post-Obs Y)

**Highest-EV remaining items**:

1. **#96 HGEMM SASS hand-tune** — the only remaining path to break
   HGEMM's 46% TC util ceiling. NCU showed `stall_math_throttle = 35.46`
   on hgemm_16warp = HMMA queue pressure. Needs CuAssembler-level
   instruction reordering. **High effort, high reward.**

2. **#100 FA pipeline coalescing** — `load_coalesce_bytes = 16` (half of
   baseline's 31) even after pad2. Cheap diagnostic, modest gain
   (estimated 1.05-1.10×).

3. **#86 Persistent grid + cooperative** — was planned 1.15×. NCU
   data shows kernels are no longer memory-throttled (post-padding),
   so this win may not materialize. Needs re-measurement.

4. **#87 Streaming K-split** — 1.5× for skinny GEMM. Regime-specific,
   needs dispatch logic.

5. **#90 SASS instruction histogram** — documentation, complements NCU.

6. **#91/#92 Register analyzer + measured roofline** — documentation.

7. **#93/#94/#95 Cymatic + autotuner** — lower priority.

**No more obvious smem-conflict wins remaining.** All canonical kernels
are now padded. The pattern has been exhausted across the codebase.

### Completed since session start

- 4 commits (ef77e0d, 569c7f8, 37bc94a, f2100ac)
- 7 issues closed
- 5 new observations (U-Y) in `gpu_reflections.md` (+490 lines)
- 5 new canonical kernels (`*_pad*.cu`)
- NCU profiling harness + 8 measurement files in `results/ncu/`

**Open issues from previous sprint plan still valid**:
- #89 NCU profiling harness — **DONE this session** (close)
- #90 SASS instruction histogram — still useful, complements NCU
- #91 Register pressure analyzer — explains FA occupancy gap (16.4% vs
  expected 25%)
- #92 Measured roofline — now feasible (NCU gives DRAM BW directly)
- #93/#94 Cymatic — unchanged
- #95 Tile autotuner — unchanged
- #96 SASS hand-tuning — unchanged

## Original Next Sprint Plan (issues #84-96, partially invalidated)

**Filed 2026-05-07. Multiplicatively could close ~3.5× of HGEMM gap and
~5-6× of FA gap to cuBLAS / FA-2 SOTA.**

### Suggested phasing

**Phase A — diagnostics (start here, ~3 days tooling)**
- #89 NCU profiling harness — capture L1/L2/DRAM hit rates per kernel
- #91 Register pressure / spill analyzer — audit `cuobjdump --dump-resource-usage`
- #90 SASS instruction histogram — comparative per-kernel HMMA/LDSM/FFMA breakdown

With these, Phase A enables real (not estimated) gap analysis and validates
or refutes hypotheses in `docs/comparison_to_sota.md`.

**Phase B — biggest single win (next deep-work session, ~1 week)**
- #84 Split-Q parallelism for Flash Attention (🔥 high-priority, 3× at small seq)
  - Two strategies: split-Q within block (cheap), split-Q across blocks (proper)
  - Single highest-EV remaining optimization in the project
  - Closes most of the FA gap to FA-2

**Phase C — pipeline + persistent (~1 week)**
- #85 4-stage cp.async pipeline for HGEMM (1.5×)
- #86 Persistent grid + cooperative dispatch (1.15× + unlocks 84/87)
- #87 Streaming K-split with cross-block reduction (1.5× for skinny)

**Phase D — fill-out (~1 week)**
- #88 XOR-swizzled smem (eliminate `+8` padding tax) (1.05×)
- #92 Measured roofline (depends on 89)
- #95 Tile-size autotuner / dispatch (1.10×)
- #96 Hand-tuned SASS via CuAssembler (1.10×)
- #93 Cymatic mode optimization search
- #94 Real-kernel cymatic integration with FA (depends on 93)

### Compound speedup targets

If Phase B + C land:
- HGEMM 18.3% peak → **~50% peak** (from 4× to 1.5× behind cuBLAS)
- FA 6.6% peak → **~30% peak** (from 7-8× to 2-3× behind FA-2)

If Phase D also lands:
- HGEMM → **~65% peak** (close to cuBLAS, ~1.2× gap)
- FA → **~40% peak** (within striking distance of FA-2)

These are upper-bound targets; real compounding is usually less than
multiplicative. Realistic outcome: 50-70% of the ideal compound gain.

## Cymatic Memory Layout (this session, post-tutorial)

**Chladni memory layout study — fully tested on GPU (audit Tier 12: treated as regular phase4 kernel)**

- `scripts/cymatic/cymatic_mapping.R` — Bessel zeros, Chladni mode field, region
  flood fill (O(N²) preallocated queue), radial-then-angular ordering,
  linear address assignment
- `scripts/cymatic/cymatic_visualize.R` — 4-panel ggplot output
- `scripts/cymatic/cymatic_analyze.R` — static locality metric (note: predicts
  circular-sweep loss; real bench shows tie/win — metric flaw documented)
- `phase4/cymatic/gen_cymatic_data.R` — emits `perm.bin` + 15 traces
- `phase4/cymatic/bench.cu` — gather-sum kernel, median + auto-iters
- `phase4/cymatic/Makefile` + `README.md` + `results/{256,512,1024,2048}.txt`
- `docs/cymatic_memory_mapping.md` — theory + bench results
- `docs/figures/cymatic/cymatic_*.png` — visualizations (mode 4,3 and 6,4)

**Headline findings (RTX 3070 Ti, GRID=2048² DRAM):**
- `radial_mid_pi6` (sector midline): **1.53×** cymatic wins
- `radial_bnd_pi4` (sector boundary): **0.54×** = 1.85× row wins
- `circular_r030`: 1.38× cymatic wins (intra-band θ-ordering matters)
- `rowmajor_full`: 0.66× = 1.51× row wins (its native pattern)
- random/biased/polar: 1.00-1.07× (ties)

**Insights captured in Observation T (`docs/gpu_reflections.md`)**:
1. Layout amplifies its mode geometry — best win 1.53×, worst loss 1.89×,
   symmetric magnitudes
2. Cache regime matters — must measure at DRAM scale (2048² / 13 MB);
   smaller buffers are L2-resident, layout differences hidden
3. Static locality metric was wrong about circular sweeps — region-level
   ordering gives tangential locality the per-pair metric missed.
   **Always validate with real GPU bench.**

**Methodology fixes applied during session**:
- R flood fill `c(qi, ni)` → preallocated queue (51s for 2048² vs hung)
- Mean-of-5 timing → median-of-11 + auto-scaled iters (≥5 ms/kernel)
- Bytes counted excludes index buffer (sequential, L1-amortized)

## Closed in deprioritized cleanup pass

| # | resolution |
|---|---|
| 14 | Tutorial series. **All 6 chapters complete** in full prose (~20K words). 01 SASS, 02 GEMM, 03 INT8, 04 Pipelining, 05 Flash Attention, 06 Four Laws. Each chapter cross-checked against repo files and benchmark numbers. |
| 18 | Static smem max (hard); reg budget violation (would force spills); Observation S occupancy logic. Of the three, only the first is a hard blocker (compile error). The other two are performance penalties — closure stands but the framing was overconfident. Counter-example file kept (does not build by design). |
| 80 | LDSM is 4.6% of cubin instructions; warp scheduler at 12 warps already hides the conflicts; Stage 3 padding (alternative mechanism) was tied with v2 within ±3%; XOR swizzle would address a non-bottleneck. |

---

## Files Map (current canonical kernels)

### Flash Attention (`kernels/attention/flash_attention/`)

- `flash_attn_br16_v2.cu` — **canonical** (smem_work eliminated, 24 KB, 3 blocks/SM)
- `flash_attn_br16_v2_pipeline.cu` — **best perf** (cp.async double-buffer, 40 KB, 2 blocks/SM)
- `flash_attn_br16_regpv.cu` — previous canonical (kept as baseline)
- `flash_attn_br16_regpv_lean.cu` — stage 1 of refactor (lean state)
- `flash_attn_br16_regpv_lean_qcache.cu` — stage 1b (Q register cache)
- `flash_attn_br16_regpv_full.cu` — stage 3 (KV/W padding, marginal vs v2)
- `flash_attn_br16_regpv_pad.cu` — counter-example (Observation O)
- `flash_attn_v2_persistent.cu` — persistent grid + nosmem (Observation Q, loses at 12 warps)
- `bench_v2_variants.cu` — sweep harness for v2 / v2_pipeline / v2_persistent
- `bench_br16_regpv_pad.cu` — sweep harness covering all regpv variants

### Cross-Attention (`kernels/attention/cross_attention/`)

- `cross_attn.cu` — original (use at small KV, e.g. CLIP-77)
- `cross_attn_v2.cu` — **nosmem variant** (use at typical sizes)
- `bench_v2.cu` — sweep harness with regime crossover analysis

### ResBlock (`kernels/convolution/resblock/`)

- `bench.cu` — original (uses slow conv2d_nhwc, kept as baseline)
- `bench_implicit.cu` — **canonical** (uses implicit_gemm_conv, 7.01× speedup)

### Documentation

- `docs/fragment_shfl_reductions.md` — **NEW** reusable pattern reference
- `docs/gpu_reflections.md` — Observations O, P, Q, R, S added this session
- `docs/tutorial/` — **NEW** complete 6-chapter tutorial series (~80 KB, all chapters full prose)

### R Helper Scripts

- `scripts/audit/ldmatrix_conflicts.R` — **NEW**: bank-conflict + occupancy tradeoff calculator
  - `ldmatrix_x4_conflict()`, `find_min_pad()`, `flash_attn_smem()`, `pad_tradeoff()`
  - CLI: `Rscript scripts/audit/ldmatrix_conflicts.R --flash-attn`
- (8 prior scripts, see "R Helper Suite" archive section below)

---

## Next-Step Suggestions (when picking up)

### A. Push beyond the 11.5 TFLOPS plateau (high value, high effort)

The v2_pipeline kernel achieves 11.5 TFLOPS = 6.6% of FP16 TC peak. After
this session's deprioritized-cleanup pass, the simple geometry knobs are
empirically exhausted on this kernel:

1. ~~Bc=128 with v2's smem savings~~ — **measured this session, lost 7%** (Observation S). Bigger tile costs occupancy 12 → 8 warps; gains from larger inner-K do not compensate.
2. ~~XOR swizzle~~ — **closed #80**: addresses a non-bottleneck (LDSM is 4.6% of cubin, hidden at 12 warps).
3. **Split-Q parallelism within a block** — untried. Currently 4 warps share Q rows of a single block. Splitting Q across blocks and reducing across blocks via atomicAdd or persistent grid reduction could enable 16+ warp configurations. Real-but-complex; not yet attempted.
4. **FP8 IMMA path** — sm_89+ feature, hardware-blocked on GA104 (sm_86). Cannot be pursued on this device.

The ~11.5 TFLOPS plateau appears genuinely difficult to break with naive
geometry tweaks; remaining options require structural redesign (split-Q)
or different hardware.

### B. Extend the fragment-shfl pattern to other kernels (medium value, medium effort)

The pattern in `docs/fragment_shfl_reductions.md` should generalize to:
- LayerNorm with Tensor Core matmuls (mean + variance reductions)
- GroupNorm in conv-after-conv contexts (now obsolete for ResBlock per #4 close, but applies elsewhere)
- Custom softmax variants

Pick one with measured smem-traffic-bound behavior (`LDS+STS > 100` in
cubin), apply pattern, benchmark.

### C. Complete tutorial chapters 01-05 (low value, medium effort)

Each outline in `docs/tutorial/` is detailed enough to write the full
chapter. Suggested order: 05 (FA, richest material now) → 02 (GEMM,
foundational) → 04 (pipelining) → 03 (INT8) → 01 (SASS, partly covered in
kernels/tutorial/README.md).

### D. Cross-attention regime dispatch in production (low effort)

Add a small selection helper in cross-attention call sites:
```cpp
if ((size_t)seq_kv * seq_q >= 200000) launch_v2(); else launch_baseline();
```

Tests show 1.43-1.62× win above this threshold, 0.81× loss below.

---

## Closed Previously (historical archive)

| # | Title | Resolution |
|---|-------|------------|
| 75 | BK_STRIDE coprime-to-32 rule | Works for inline-asm LDS only, incompatible with WMMA alignment. Documented. |
| 74 | conv2d benchmark binary hangs | Relative-path cubin loading fixed via `find_cubin()` helper. |
| 76 | Persistent grid kernels slower | Adaptive grid `min(96, total_tiles)` wins +10-16% medium sizes, loses at 4096³. |
| 77 | Fused B-load slower than SWIZZLE_B | Per-thread scatter < bulk transpose; counter-example retained. |
| 66 | LDSM B-pack for sparse INT8 GEMM | Closed (previous session). |
| 67 | Migrate 23 bench files to BenchDriver | Closed (previous session). |

## R Helper Suite (legacy + new)

| Script | Purpose |
|--------|---------|
| `scripts/model/analyze_smem_layout.R` | INT8-sparse-GEMM bank conflict analyzer |
| `scripts/model/find_optimal_smem_layout.R` | Optimal INT8 B-smear layout |
| `scripts/audit/track_prmt_reduction.R` | Cubin PRMT/LDS/MMA counter |
| `scripts/model/occupancy_calc.R` | GA104 occupancy calculator |
| `scripts/model/perf_model_panel.R` | Roofline + memory ceiling analysis |
| `scripts/model/config_optimizer.R` | Grid-search BM/BN/BK/warp configs |
| `scripts/model/pipeline_balance.R` | Compute-vs-load overlap model |
| `scripts/model/kernel_dashboard.R` | Combined dashboard |
| `scripts/audit/ldmatrix_conflicts.R` | **NEW** generic ldmatrix.x4 + FA smem + pad-vs-occupancy tradeoff |

## Key Findings (legacy, still valid)

1. **Coprime-to-stride/4 rule**: For inline-asm `ld.shared.b32`, `stride_bytes / 4` must be odd. WMMA/ldmatrix cannot use it (16-byte alignment forces even). For WMMA, use **fragment-shfl reductions** (this session, insight 1) instead.
2. **Persistent grid sweet spot**: Medium tile counts (128-512) at 8 warps/SM. At 12 warps/SM the wins disappear (this session, insight 3).
3. **Fused scatter < bulk transpose**: cp.async row-major→row-major + bulk SWIZZLE_B beats per-thread scatter.
4. **CWD matters for benchmarks**: Relative cubin paths fail silently. Use `find_cubin()` or absolute paths.

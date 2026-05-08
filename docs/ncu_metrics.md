# Measured NCU Metrics — per-kernel cache + stall breakdown

> **Status**: first-pass measurement complete (2026-05-08).
> Source CSV: `results/ncu/all.csv`. Harness: `scripts/ncu_profile_all.sh`.
> Re-run after any kernel change.
>
> Headline finding: **Flash Attention is smem-traffic-bound, not
> HMMA-bound**. See `gpu_reflections.md` Observation U for full analysis.

## Method

For each canonical kernel, with `--launch-skip 5 --launch-count 1`,
captured 15 metrics. All metric names validated against
`ncu --query-metrics --chip ga104` before measurement.

| short label | NCU metric | what it measures |
|---|---|---|
| `occupancy_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | active warps vs 32/SM ceiling |
| `tc_util_pct` | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | Tensor Core busy fraction |
| `l1_hit_pct` | `l1tex__t_sector_hit_rate.pct` | L1 sector hit rate (all lookups) |
| `l2_hit_pct` | `lts__t_sector_hit_rate.pct` | L2 sector hit rate (all lookups) |
| `dram_read_bw` | `dram__bytes_read.sum.per_second` | measured DRAM read BW |
| `dram_write_bw` | `dram__bytes_write.sum.per_second` | measured DRAM write BW |
| `load_coalesce_bytes` | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | bytes/sector on global loads (32 = perfect) |
| `stall_long_sb` | `…long_scoreboard…` | DRAM/L2 latency stall (memory) |
| `stall_short_sb` | `…short_scoreboard…` | smem/L1 latency stall |
| `stall_wait` | `…wait…` | dependency wait (HMMA S08, FFMA chains) |
| `stall_mio` | `…mio_throttle…` | memory IO unit throttle (smem traffic) |
| `stall_lg_throttle` | `…lg_throttle…` | local/global memory throttle |
| `stall_barrier` | `…barrier…` | `__syncthreads()` wait |
| `stall_math_throttle` | `…math_pipe_throttle…` | FFMA/INT/FP pipe oversubscription |
| `stall_tex_throttle` | `…tex_throttle…` | tex unit throttle |

DRAM peak (GA104): **608 GB/s**. FP16 TC peak: **174 TFLOPS**.
32 warps/SM ideal, 100 KB max smem/SM (50 KB cliff for 2 blocks/SM).

Stall units are "average per-issue cycles" — fraction of the
instruction issue stream blocked by that reason. Higher = worse.
A stall of 35 means the pipe is waiting that many cycles per issue
on average.

## Headline metrics across canonical kernels

| kernel | occ% | TC% | L1 hit% | L2 hit% | DRAM rd GB/s |
|---|---:|---:|---:|---:|---:|
| FA v2 baseline (seq=1024, b=8, h=8) | 24.2 | 12.1 | 10.8 | 91.4 | 11.5 |
| FA v2 pipeline | 16.4 | 13.9 | 9.3 | 91.6 | 13.1 |
| FA v2 persistent | 17.1 | 11.2 | 11.1 | 89.4 | 10.6 |
| FA regpv (legacy) | 24.3 | 8.6 | 23.1 | 94.9 | 9.1 |
| HGEMM 16-warp (4096³) | 65.1 | **46.3** | 1.6 | 76.4 | 168.3 |
| HGEMM 16-warp+epi | 65.4 | 26.1 | 0.3 | 64.9 | 94.7 |
| HGEMM 256x128 | 33.3 | 26.1 | 2.9 | 74.1 | 50.9 |
| Sparse INT8 GEMM (4096³) | 65.4 | 14.0 | 83.4 | 77.7 | 117.4 |
| Cross-attn v2 (1024 q, 256 kv, h=8) | 21.0 | 10.5 | 27.1 | 87.0 | 19.0 |
| ResBlock implicit GEMM (SD UNet 320ch) | 14.5 | 3.2 | 54.4 | 98.2 | 3.5 |

## Stall histograms (per-issue cycles)

| kernel | wait | mio | short_sb | long_sb | math_thr | barrier | lg_thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| FA v2 baseline | 1.08 | **7.87** | 2.39 | 1.85 | 1.12 | 1.55 | 0.19 |
| FA v2 pipeline | 0.93 | **6.89** | **5.41** | 0.19 | 1.44 | 2.35 | 0.16 |
| FA v2 persistent | 1.08 | **4.66** | 2.37 | 1.97 | 0.97 | 0.92 | 0.15 |
| FA regpv (legacy) | 1.48 | **4.90** | **5.47** | 2.02 | 0.82 | 1.70 | 0.10 |
| HGEMM 16-warp | 2.63 | 3.47 | 2.21 | 1.01 | **35.46** | 5.33 | 0.09 |
| HGEMM 16-warp+epi | 2.36 | **20.98** | **16.82** | 1.72 | 5.91 | **17.32** | 0.01 |
| HGEMM 256x128 | 3.29 | **24.01** | 6.03 | 1.03 | 15.12 | **16.06** | 0.15 |
| Sparse INT8 GEMM | 2.50 | **10.71** | 1.54 | **10.50** | 0.45 | 2.99 | 1.06 |
| Cross-attn v2 | 1.07 | **6.18** | 2.75 | 2.60 | 0.99 | 1.05 | 0.47 |
| ResBlock implicit | 2.40 | 0 | 1.06 | **5.13** | 0.24 | 0.56 | 0 |

Bold = ≥ 4.5 (visually dominant stall in that row).

## Per-kernel diagnosis

### Flash Attention v2 pipeline (current canonical)

- **Bottleneck: smem traffic.** `stall_mio + stall_short_sb = 12.3` — far
  larger than every other stall combined.
- HMMA `stall_wait = 0.93` — **HMMA S08 NOT the wall**. Long-held
  assumption refuted.
- L2 hit 91.6% → KV reuse is excellent, NOT DRAM-bound.
- DRAM read BW 13 GB/s vs 608 GB/s peak (2.1%) — bandwidth slack is huge.
- Coalescing 16 byte/sector — half perfect. Lost since baseline (31).
  Pipeline cp.async path is somehow degrading load coalescing.
  **Worth investigating**.
- Occupancy 16.4% — lower than baseline's 24.2%. Two blocks/SM trade
  cost. Pipeline still wins on time because cp.async overlaps smem
  fills with HMMA.

**Right structural fix**: reduce smem traffic. Candidates:
1. XOR-swizzled smem layout (#88) — promote, was previously deprioritized
2. Extend fragment-shfl pattern to load path
3. Find why pipeline lost coalescing relative to baseline

**Wrong fix**: split-Q (#84) — addresses SM starvation; we saturate 22×.

### Flash Attention regpv (legacy)

- Highest occupancy (24.3%) but lowest TC util (8.6%) of FA variants.
- Confirms occupancy alone is not predictive — kernel structure matters.
- Same smem-bound profile as v2 (stall_mio 4.90, short_sb 5.47).

### HGEMM 16-warp

- **Bottleneck: FFMA pipe oversubscription.** `stall_math_throttle = 35.46`
  — far the dominant stall.
- TC util 46.3% — actually good (much higher than the 18.3% throughput
  number suggested). The gap is the FFMA pipe stalling everything else.
- L1 hit 1.6% — kernel routes everything through smem (expected for
  GEMM at this tile size).
- L2 hit 76.4% — column buffer reuse working as designed.
- DRAM 168 GB/s — moderate (28% of peak) but compute-bound.

**Right structural fix**: hunt and reduce the FFMA chain. Probably in
accumulator scaling or epilogue. Closing this could push TC util from
46% → 70-80%.

**Wrong fix**: more pipelining (#85) — TC is already at 46% busy.

### HGEMM 16-warp+epi

- **Triple stall hot spots**: `stall_mio=20.98`, `stall_barrier=17.32`,
  `stall_short_sb=16.82`.
- TC util drops 46% → 26% vs plain 16-warp.
- The "fused epilogue" routes accumulator through smem with extra syncs.
  Cost > savings.

**Action**: don't use this variant for production until rewritten.
Plain `hgemm_16warp` is the canonical at this point. Or apply the
fragment-shfl pattern (Observation P) to the epilogue.

### HGEMM 256x128

- 33% occupancy (lowest of HGEMM family) due to bigger tiles.
- `stall_mio=24.01` and `stall_barrier=16.06` dominate.
- 26% TC util vs plain 16-warp's 46% — bigger tiles lose here.
- Confirms Observation S: bigger tiles cost occupancy more than they
  save.

### Sparse INT8 GEMM (2:4)

- `stall_long_sb=10.50` — DRAM-latency bound, unique among kernels here.
- L1 hit 83.4% — anomalously high. Probably index/mask buffers staying
  in L1. Worth confirming.
- TC util 14.0% despite 1.39× dense-equiv speedup — sparse instructions
  IMMA.SP have lower achievable density.
- `stall_mio=10.71` also large. Combined memory + smem path are the
  twin walls.

**Action**: investigate L1 hit anomaly (load_coalesce_bytes = 2.12
suggests the index buffer is not coalesced — that may be feeding L1).

### Cross-attention v2

- Profile mirrors FA v2 pipeline: smem-bound.
- L2 hit 87% — good KV reuse.
- Coalescing 28.8 byte/sector — much better than FA pipeline's 16.

### ResBlock implicit GEMM

- TC util **3.2%** — terrible despite 7.01× speedup headline.
- Workload (b=1, 320ch, 32×32) is so small it can't fill the GPU.
- Occupancy 14.5%.
- `stall_long_sb=5.13` + `stall_wait=2.40` dominate — small problem,
  exposed memory latency.
- Headline speedup is from reducing 9× input re-reads (Observation R),
  not from utilization.
- **Implication**: real speedup ceiling on this config is hardware-
  limited by problem size. Bigger workloads (training vs inference)
  would scale better.

## Cross-reference to gap analysis

`docs/comparison_to_sota.md` per-factor accounting was estimated.
Replacements after this measurement:

| factor | est. before | measured |
|---|---|---|
| HGEMM Tensor Core util | "~40%" | **46.3%** ✓ close |
| HGEMM bottleneck | "TC density" | **FFMA throttle** ✗ wrong |
| FA Tensor Core util | "~10-15%" | **13.9%** ✓ close |
| FA bottleneck | "HMMA S08 stalls" | **smem traffic** ✗ wrong |
| FA L2 hit rate | "60-80% est" | **91.6%** ✓ better than estimated |

Two of five estimates were wrong about the bottleneck mechanism, even
when the magnitude was close. **Mechanism is what determines the right
fix.** Update `comparison_to_sota.md` to mark these `(measured)` and
revise the optimization plan.

## Reprioritization for next sprint

Before NCU: top candidates were #84 (split-Q FA), #85 (HGEMM
4-stage pipeline), #86 (persistent grid).

After NCU:
1. **NEW: Hunt FFMA chain in HGEMM 16-warp** — `stall_math_throttle=35.46`.
   Close most of the 46% → 70-80% TC util gap. No issue yet, file one.
2. **#88 XOR-swizzled smem** — promote. Directly attacks the dominant
   FA stall (`stall_mio + stall_short_sb`).
3. **NEW: Extend fragment-shfl pattern to FA load path** — Observation P
   already validated the idea on the score buffer; load path is a
   second use.
4. **#84 split-Q** — close as not-planned for trained-attention shapes,
   OR refile targeting decoding (b=1, seq_q=1) regime where SMs DO
   starve.
5. **#85 4-stage pipeline** — re-evaluate after FFMA hunt completes.
   Pipelining loads won't help if FFMA is the wall.

## Methodology notes

- WSL2 + Windows host: NCU counters require enabling on the host side
  (NVIDIA Control Panel → Developer Settings → Manage GPU Performance
  Counters → "Allow access to all users"). After enable, reboot.
  WSL-side `ncu` binary then works without sudo.
- All metric names validated offline via `ncu --query-metrics --chip ga104`
  before runtime.
- Single-launch capture is noisy. For publication-quality numbers, re-run
  with `--launch-count 5` and average.
- Stall metrics are "per-issue cycles" — relative to instruction issue
  rate, not wall time. Comparing across kernels of different sizes is
  valid as long as the metric is interpreted as "fraction of the issue
  stream blocked by this reason".

## Files

- `scripts/ncu_profile.py` — single-kernel wrapper, supports `--dry-run`
- `scripts/ncu_profile_all.sh` — sweep across canonical kernels
- `results/ncu/all.csv` — raw measurement output
- `results/ncu/smoke.csv` — single-kernel smoke test (FA v2 pipeline only)

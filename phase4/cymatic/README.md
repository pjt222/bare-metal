# Cymatic Memory Mapping — GPU Benchmarks

> Empirical measurement of whether the [cymatic memory layout](../../docs/cymatic_memory_mapping.md)
> actually outperforms row-major on real GPU hardware. Result: **yes for some
> access patterns, no for others, geometry-dependent**. The layout is a real
> tradeoff, not a free win.

## What this benchmarks

Two memory layouts of the same logical data:

- **Row-major**: cells inside the disc indexed in raster order (i ascending, j ascending).
- **Cymatic**: cells permuted to the layout produced by `scripts/cymatic/cymatic_mapping.R`,
  ordered by (centroid_r, centroid_θ) over Chladni-mode antinode regions.

Both layouts hold the same 32-bit float values; only the physical positions
differ. The benchmark runs a gather kernel:

```cuda
__global__ void gather_sum(const float *data, const int *idx,
                           float *out, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (int it = 0; it < iters; ++it)
        for (int k = tid; k < n; k += stride)
            s += data[idx[k]];
    if (tid == 0) atomicAdd(out, s);
}
```

For each access trace, we feed the same logical cell sequence as two index
buffers — `idx_row[t] = trace[t]` for the row layout, `idx_cym[t] =
perm[trace[t]]` for the cymatic layout — and time both. Same kernel, same
trace, same data values → bandwidth difference is pure layout.

## How to run

```bash
# 1. Generate inputs (perm.bin + traces.bin) for a 1024×1024 grid, mode (6, 4)
make gen GRID=1024

# 2. Build CUDA bench
make

# 3. Run (defaults: 200 iters/run, 5 warmup, 11 runs, median reported)
./bench_cymatic

# Or sweep grid sizes
make sweep
```

Output: per-trace table with row_ms, row_GB/s, row_eff%, cym_ms, cym_GB/s,
cym_eff%, speedup (= row_ms / cym_ms; > 1 ⇒ cymatic wins).

Captured runs land in the centralised `results/cymatic/grids/` tree at
the repo root (`grid<N>_results.txt` for N = 256, 512, 1024, 2048).

## Key result: cymatic locality is angle-dependent

For mode (n=6, m=4), the angular sectors have midlines at θ = k·π/6 (where
`cos(6θ) = ±1`) and boundaries at θ = π/12 + k·π/6 (where `cos(6θ) = 0`).
A radial trace at a sector midline stays within one sector through all
m=4 radial bands, hitting cymatic addresses in a near-contiguous block. A
trace at a sector boundary sits exactly on the nodal line between two
opposite-sign regions, so adjacent (i, j) cells in the trace map to
entirely different region address ranges. Worst case for the layout.

Measured speedup at GRID=2048 (13 MB buffer, fully DRAM):

| trace | speedup | interpretation |
|---|---|---|
| `radial_mid_pi6` (θ=π/6, midline) | **1.53×** | cymatic wins |
| `radial_mid_0` (θ=0, midline) | 1.01× | tie — row layout already coalesces (cells in single row) |
| `radial_bnd_pi4` (θ=π/4, boundary) | **0.54×** | cymatic loses by 1.85× |
| `radial_bnd_5pi12` (θ=5π/12, boundary) | **0.53×** | cymatic loses by 1.89× |
| `circular_r030` (small radius) | **1.38×** | cymatic wins (radial-band scan) |
| `circular_r060` (large radius) | 1.12× | mild cymatic win |
| `polar_tile_pi6` (midline-centered) | 0.98× | tie |
| `polar_tile_pi4` (boundary-centered) | 1.03× | tie |
| `radial_bias_07` (random gather, r₀=0.7) | 1.07× | tie |
| `random` (uniform shuffle) | 1.03× | tie |
| `rowmajor_full` (sequential native row scan) | **0.66×** | row layout wins by 1.51× |
| `colmajor_full` (sequential column scan) | 0.98× | tie (both bad) |

Reading: **the layout amplifies its mode geometry**. Workloads that align
with a sector midline are strongly accelerated; workloads that graze a
nodal line are strongly decelerated; everything else is neutral. The
worst slowdown (1.89×) is comparable in magnitude to the best speedup
(1.53×). For a workload with a fixed access pattern, this is meaningful.
For a generic workload, it's a wash.

## Why circular sweeps win (the R analysis was wrong)

The R locality analysis (`scripts/cymatic/cymatic_analyze.R`) predicted that
circular sweeps at fixed r should hurt cymatic locality because
"adjacent θ → different angular sectors → address jumps". The benchmark
contradicts this prediction: circular sweeps are tied or favor cymatic.

The reason: cymatic regions are ordered as (centroid_r, centroid_θ) —
all regions in one radial band sit in a contiguous address range, with
addresses sorted by θ within the band. A circular trace at fixed r stays
in one radial band the entire time and scans through θ-sorted regions →
**addresses are roughly monotone**, not random. The intra-band ordering
gives it locality even tangentially.

This is a real-system finding the static metric missed. The CUDA bench
catches it; the R metric does not.

## Result table across grid sizes (speedups, robust patterns only)

| Pattern | 256² | 512² | 1024² | 2048² |
|---|---|---|---|---|
| `radial_bnd_pi4` | 0.98× | 0.99× | **0.63×** | **0.54×** |
| `radial_bnd_5pi12` | 0.99× | 0.96× | 0.99× | **0.53×** |
| `radial_mid_pi6` | 0.98× | 0.99× | 1.00× | **1.53×** |
| `circular_r060` | 1.02× | 0.97× | **1.86×** | 1.12× |
| `circular_r030` | 0.97× | 0.98× | 0.90× | **1.38×** |
| `rowmajor_full` | 0.78× | 1.40× | 1.09× | **0.66×** |

Buffer sizes: 256² = 0.2 MB (L1/L2), 512² = 0.8 MB (L2), 1024² = 3.3 MB
(L2 boundary), 2048² = 13 MB (DRAM). Wins and losses sharpen at DRAM
scale because the cache no longer hides locality differences.

The smaller grids show mostly ties because the entire buffer fits in
L2 and post-warmup all accesses are L2 hits regardless of layout. The
2048² results are the "true" measurement.

## Methodology

- Median of 11 measured runs (after 5 warmup) per (trace, layout)
- Iters per run auto-scaled: small traces get 5–25× more iters so each
  measured kernel runs ≥1 ms (above ~10 μs CUDA event-timer noise)
- Bytes counted as data only (4 × n × iters); index buffer accesses are
  sequential and amortized via L1, excluded to keep cym/row honest
- Bandwidth `> 100% of peak` indicates cache hits, not measurement error
  — the buffer is reused across iters, so post-warmup the trace is
  L2-resident; reported "efficiency %" should be read as L1+L2+DRAM
  aggregate throughput, not pure DRAM
- Tested on RTX 3070 Ti Laptop (GA104, sm_86, 46 SMs, 4 MB L2, 608 GB/s DRAM peak)

## Files

- `gen_cymatic_data.R` — generates `perm.bin` + `traces.bin` from R math.
  Sources `../../scripts/cymatic/cymatic_mapping.R` and `cymatic_analyze.R`.
- `bench_cymatic.cu` — CUDA gather bench with median + scaled iters.
- `Makefile` — `make`, `make gen`, `make run`, `make sweep`, `make clean`.
- `results/cymatic/grids/grid{256,512,1024,2048}_results.txt` — captured benchmark output.

## Honest assessment

Cymatic memory mapping is **not a universal speedup**. At DRAM scale
(2048², 13 MB buffer) on RTX 3070 Ti:

- **Wins** when the access pattern aligns with mode geometry —
  midline radial sweeps (1.5×), small-radius circular (1.4×). This
  matches workloads where the data has true rotational/radial
  structure: polar warps, FFT butterflies in radial decomposition,
  attention with rotational position bias.
- **Loses** when the pattern grazes mode nodal lines (boundary radial
  sweeps, ~1.9× slowdown) or matches row-major native order
  (sequential full scans, ~1.5× row-layout advantage).
- **Indifferent** for random gather, polar tiles, and biased samples.

For a fixed workload with known geometry, this layout is a real tool.
For a workload with unknown or mixed access patterns, row-major is
safer.

The benchmark proves the cymatic mapping is a measurable physical
phenomenon on real hardware, not just an analytical curiosity. It also
proves the layout is conditional, not universal.

## Possible next steps

1. **Mode optimization**: given a workload's known access pattern,
   search over (n, m, α) modes to maximize speedup. The search space
   is tiny (small integers + 1–2 reals), the metric is reproducible.
2. **Hierarchical cymatic**: outer mode for coarse partition, inner
   mode within each region for fine layout. Might capture multi-scale
   patterns.
3. **Real-kernel integration**: replace `phase3/flash_attention/` Q/K/V
   buffer layout with cymatic and measure end-to-end FA throughput.
   The QK^T pattern has rotational structure (each query attends across
   all keys); could match midline radial alignment.
4. **L2 persistence**: pin cymatic regions in L2 via `cudaAccessPolicyWindow`
   so the layout's locality benefit is amplified when the working set
   exceeds L2.

## Cross-references

- `docs/cymatic_memory_mapping.md` — full theory and R-side analysis
- `scripts/cymatic/cymatic_mapping.R` — region computation
- `scripts/cymatic/cymatic_analyze.R` — static locality metric (note: predicts
  some patterns wrong vs measured GPU bench, see "circular sweeps" above)

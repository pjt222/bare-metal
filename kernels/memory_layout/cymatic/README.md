# kernels/memory_layout/cymatic — Chladni-pattern memory layout (GPU benchmark)

> Empirical measurement of whether the [cymatic memory layout](../../../docs/cymatic_memory_mapping.md)
> actually outperforms row-major on real GPU hardware. Result: **yes for some
> access patterns, no for others, geometry-dependent**. The layout is a real
> tradeoff, not a free win.

## What this benchmarks

Two memory layouts of the same logical data:

- **Row-major**: active cells indexed in raster order (i ascending, j ascending).
- **Cymatic**: cells permuted to the layout produced by `scripts/cymatic/cymatic_mapping.R`,
  ordered by (centroid_r, centroid_θ) over Chladni-mode antinode regions.

Both layouts hold the same 32-bit float values; only the physical positions
differ. The benchmark runs a gather kernel:

```cuda
__global__ void gather_sum(const float *data, const int *idx,
                           float *out, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = 0.0f;
    for (int it = 0; it < iters; ++it)
        for (int k = tid; k < n; k += stride)
            s += data[idx[k]];
    out[blockIdx.x] = block_reduce_sum(s);
}
```

The sampling grid still spans the full disc bounding box, and the benchmark
now supports three domain/layout variants:

- **`DOMAIN=disc`** — original in-circle disc
- **`DOMAIN=square`** — inscribed square, with the outer disc ring dropped
- **`DOMAIN=overlayed`** — full disc support, but with the square-domain core
  ordered first and the remaining disc-only ring ordered afterward

The `--overlay` visualization comparing the disc and square masks is written to
`docs/figures/cymatic/cymatic_domain_overlay.png`.

For each access trace, we feed the same logical cell sequence as two index
buffers — `idx_row[t] = trace[t]` for the row layout, `idx_cym[t] =
perm[trace[t]]` for the cymatic layout — and time both. Same kernel, same
trace, same data values → bandwidth difference is pure layout.

## How to run

```bash
# 1. Generate the original disc-domain inputs
make gen GRID=1024 DOMAIN=disc

# 2. Generate the inscribed-square inputs
make gen GRID=1024 DOMAIN=square

# 3. Generate the composite overlayed inputs
make gen GRID=1024 DOMAIN=overlayed

# 4. Build CUDA bench
make

# 5. Run any variant (defaults: 200 iters/run, 5 warmup, 11 runs, median reported)
make run DOMAIN=disc
make run DOMAIN=square
make run DOMAIN=overlayed

# Or sweep grid sizes
make sweep DOMAIN=disc
make sweep DOMAIN=square
make sweep DOMAIN=overlayed
```

Output: per-trace table with row_ms, row_GB/s, row_eff%, cym_ms, cym_GB/s,
cym_eff%, speedup (= row_ms / cym_ms; > 1 ⇒ cymatic wins).

Captured runs land in the centralised `results/cymatic/grids/` tree at
the repo root (`grid<N>_<domain>_results.txt` for N = 256, 512, 1024, 2048).

## Key result: overlayed keeps the disc-sized working set and most of the disc-domain wins

For mode (n=6, m=4), the angular sectors have midlines at θ = k·π/6 (where
`cos(6θ) = ±1`) and boundaries at θ = π/12 + k·π/6 (where `cos(6θ) = 0`).
A radial trace at a sector midline stays within one sector through all
m=4 radial bands, hitting cymatic addresses in a near-contiguous block. A
trace at a sector boundary sits exactly on the nodal line between two
opposite-sign regions, so adjacent (i, j) cells in the trace map to
entirely different region address ranges. Worst case for the layout.

Measured speedup at GRID=2048 (fresh 2026-05-18 sweeps):

| trace | disc | square | overlayed | reading |
|---|---:|---:|---:|---|
| `radial_mid_pi6` | **1.50×** | 0.99× | **1.48×** | overlayed keeps the disc-style sector-midline win |
| `radial_bnd_pi4` | **0.70×** | **0.86×** | **0.71×** | overlayed keeps the disc-style nodal-boundary loss |
| `radial_bnd_5pi12` | **0.69×** | **0.70×** | **0.70×** | boundary loss is robust in all full-disc variants |
| `circular_r030` | 2.10× | 1.01× | **2.11×** | overlayed preserves the biggest disc-domain win |
| `circular_r060` | 1.21× | 0.97× | **1.25×** | overlayed also keeps the larger-radius circular win |
| `polar_tile_pi6` | **1.27×** | 0.93× | 1.02× | overlayed flattens this one back toward neutral |
| `rowmajor_full` | 0.67× | **0.82×** | **0.64×** | row-major still wins hardest when the full disc is active |
| `colmajor_full` | 0.83× | **1.09×** | 0.99× | overlayed removes the square-only col-major win |

Reading: **overlayed behaves much more like disc than square.** Because it
restores the full disc active set, it keeps the strongest geometry-aligned
wins (`circular_r030`, `radial_mid_pi6`, `circular_r060`) and the strong
row-major penalty. But the square-first address ordering does change some
mid-structure cases: `polar_tile_pi6` falls back to near-tie and
`colmajor_full` loses the square-domain advantage.

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

## Overlayed cross-grid table (selected patterns)

| Pattern | 256² | 512² | 1024² | 2048² |
|---|---|---|---|---|
| `radial_bnd_pi4` | 0.99× | 0.99× | 0.99× | **0.71×** |
| `radial_bnd_5pi12` | 0.99× | 0.96× | 0.98× | **0.70×** |
| `radial_mid_pi6` | 1.01× | 1.00× | 0.99× | **1.48×** |
| `circular_r060` | 1.02× | 0.98× | **1.89×** | **1.25×** |
| `circular_r030` | 0.99× | 0.99× | 0.98× | **2.11×** |
| `polar_tile_pi6` | 0.98× | **1.05×** | **1.10×** | 1.02× |
| `rowmajor_full` | 1.02× | 1.00× | 1.00× | **0.64×** |
| `colmajor_full` | **1.17×** | 1.04× | 1.03× | 0.99× |

Active buffer sizes for the overlayed domain: 256² = 0.20 MB, 512² = 0.82 MB,
1024² = 3.28 MB, 2048² = 13.16 MB. Like the disc domain, overlayed only
really separates from row-major once the working set pushes beyond L2.

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
  Sources `../../../scripts/cymatic/cymatic_mapping.R` and `cymatic_analyze.R`.
- `bench.cu` — CUDA gather bench with median + scaled iters.
- `Makefile` — `make`, `make gen`, `make run`, `make sweep`, `make clean`.
- `results/cymatic/grids/grid{256,512,1024,2048}_{disc,square,overlayed}_results.txt` — captured benchmark output.

## Honest assessment

Cymatic memory mapping is **not a universal speedup**. At DRAM scale on
RTX 3070 Ti:

- **Disc domain** is the more expressive variant: it keeps the large
  geometry-aligned wins (up to 2.10×) but also the clearest failures
  on nodal-boundary and row-major-native scans.
- **Square domain** is the more conservative variant: it dampens both
  the wins and the losses, leaving mostly ties plus a few modest wins
  and persistent nodal-boundary regressions.
- **Overlayed domain** preserves the disc-sized active set and most of the
  strongest disc wins, but with a square-first core ordering that flattens
  some mid-structure traces back toward neutral.
- **Indifferent** patterns remain random-gather-like traces where both
  domains stay near 1.0×.

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
3. **Real-kernel integration**: replace `kernels/attention/flash_attention/` Q/K/V
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

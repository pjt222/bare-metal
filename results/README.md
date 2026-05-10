# `results/` — captured benchmark + profiling output

Centralised home for run artefacts that aren't figures (those live in
`docs/figures/`). Two subtrees exist today; more added as new
profiling sweeps are introduced.

## Structure

```
results/
├── ncu/                     ── NCU 15-metric sweeps (audit Tier 7 entry point)
│   ├── all.csv              ── canonical roofline sweep, 10 representative kernels
│   ├── cross_check.csv      ── independent re-run for noise estimate
│   ├── 97_pad.csv           ── per-issue captures
│   ├── 99_epi_pad.csv       ──   "
│   ├── fa_pad.csv, fa_pad2.csv ── flash-attention pad-vs-no-pad
│   ├── hgemm_imad.csv       ── HGEMM IMAD-stall analysis
│   └── smoke.csv            ── pre-flight smoke run
│
└── cymatic/                 ── phase4/cymatic benchmark captures
    └── grids/               ── per-grid bench_cymatic output
        ├── grid256_results.txt
        ├── grid512_results.txt
        ├── grid1024_results.txt
        └── grid2048_results.txt
```

## Conventions

- One subdir per data domain (`ncu/`, `cymatic/`, future `bench/`,
  `regression/`, etc.).
- File names describe captured configuration, not generation date.
  Re-runs overwrite — git history holds the diffs.
- Output formats are CSV or TXT. Binary outputs (PNGs) are never
  written here; they go to `docs/figures/`.

## Generators

| Subtree | Written by |
|---|---|
| `results/ncu/all.csv`        | `scripts/profile/ncu_profile_all.sh` |
| `results/ncu/<single>.csv`   | `scripts/profile/ncu_profile.R --out results/ncu/<name>.csv` |
| `results/cymatic/grids/`     | `make -C phase4/cymatic sweep` (Makefile target) |

## Cross-references

- [`docs/ncu_metrics.md`](../docs/ncu_metrics.md) — column definitions for the NCU tables
- [`docs/roofline_measured.md`](../docs/roofline_measured.md) — interpretation of `ncu/all.csv`
- [`phase4/cymatic/README.md`](../phase4/cymatic/README.md) — what the grids/ files measure

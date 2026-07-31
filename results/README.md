# `results/` — captured benchmark + profiling output

Centralised home for run artefacts that aren't figures (those live in
`docs/figures/`). Two subtrees exist today; more added as new
profiling sweeps are introduced.

## Structure

```
results/
├── ncu/                     ── NCU 15-metric sweeps
│   ├── all.csv              ── canonical roofline sweep, 10 representative kernels
│   ├── cross_check.csv      ── independent re-run for noise estimate
│   ├── 97_pad.csv           ── per-issue captures
│   ├── 99_epi_pad.csv       ──   "
│   ├── fa_pad.csv, fa_pad2.csv ── flash-attention pad-vs-no-pad
│   ├── hgemm_imad.csv       ── HGEMM IMAD-stall analysis
│   └── smoke.csv            ── pre-flight smoke run
│
└── cymatic/                 ── kernels/memory_layout/cymatic benchmark captures
    └── grids/               ── per-grid bench output
        ├── grid256_results.txt
        ├── grid512_results.txt
        ├── grid1024_results.txt
        └── grid2048_results.txt
```

## Conventions

- One subdir per data domain (`ncu/`, `cymatic/`, `bench_regress/`, etc.).
- File names describe captured configuration, not generation date.
  Re-runs overwrite — git history holds the diffs.
- Output formats are CSV or TXT. Binary outputs (PNGs) are never
  written here; they go to `docs/figures/`.

**`bench_regress/` is the exception to all three, deliberately (#186).** It is
append-only rather than overwriting, JSONL rather than CSV/TXT, and gitignored
rather than tracked. The reason is what it is for: it is the pre-push gate's
evidence trail, and the question it answers — "which config regressed on that
rejected push, and what was the GPU doing at the time?" — is precisely the one
an overwriting store cannot answer. It is machine-specific measurement, so it
stays local; git history is not available to hold its diffs.

Being gitignored makes `git clean -xdf` its de-facto pruner, and that removes
the lot. Nothing else prunes it. It grows by roughly one line per config per
gate run, so a year of pushes is a file measured in megabytes, not gigabytes.

## Generators

| Subtree | Written by |
|---|---|
| `results/ncu/all.csv`        | `scripts/profile/ncu_profile_all.sh` |
| `results/ncu/<single>.csv`   | `scripts/profile/ncu_profile.R --out results/ncu/<name>.csv` |
| `results/cymatic/grids/`     | `make -C kernels/memory_layout/cymatic sweep` (Makefile target) |
| `results/bench_regress/gate_runs.jsonl` | `scripts/bench/bench_regress.R` — every run, appended (#186) |

## Cross-references

- [`docs/ncu_metrics.md`](../docs/ncu_metrics.md) — column definitions for the NCU tables
- [`docs/roofline_measured.md`](../docs/roofline_measured.md) — interpretation of `ncu/all.csv`
- [`kernels/memory_layout/cymatic/README.md`](../kernels/memory_layout/cymatic/README.md) — what the grids/ files measure

---
title: "scripts/bench/bench_regress.R: parser matches wrong kernel row, working-dir bug, .py refs"
labels: ["bug", "tooling"]
status: "closed by Tier 9 (fix in commit a50cf5c-successor); residual: re-baseline igemm_sparse_tiled"
---

## Status

**Resolved by Tier 9** (parser-repair commit). All three issues
fixed. Residual: one real CUDA 13.2 regression in
`igemm_sparse_tiled` 2048³ (-21% vs CUDA 12.8 baseline) which is
out of scope for this issue — see Obs HH for the underlying IMMA
stall-count change. Action: re-baseline that single entry on
CUDA 13.2 with a `recorded_date` bump or accept the regression as
known and skip in the hook.

## Problem (historical)

`scripts/bench/bench_regress.R` (the perf-regression checker invoked
by the pre-push hook) is currently non-functional. Three issues
surfaced when the post-Tier-5 audit fixed the pre-push hook to
actually invoke the script (it had been silently skipping for ages
because it looked for `scripts/bench_regress.py`, which never
existed).

## Issues

### 1. Wrong kernel row parsed (silent false regressions)

`run_benchmark()` parses the first `<X> ms ... <Y> GFLOPS` line out of
the bench's stdout. But every multi-kernel `bench.cu` (hgemm, igemm,
etc) prints results for several kernels in order — naive first, the
"good" variant later. Example, `kernels/gemm/hgemm/bench`:

```
hgemm_wmma (naive 32x32)        2.437 ms    7048.54 GFLOPS    ← parsed
hgemm_tiled (128x128 smem epi)  1.383 ms   12421.31 GFLOPS
...
hgemm_16warp (128x128 2blk/SM)  0.536 ms   32022.48 GFLOPS    ← actual baseline
```

`docs/baselines.json` records the 16warp row (31,910 GFLOPS); parser
returns the naive row (~7,000 GFLOPS); regression check reports a
~78% drop. **All 7 baseline kernels report false regressions** for
this reason.

Fix needed: bind each kernel-path key in `baselines.json` to a regex
or label substring (e.g. `"hgemm_16warp"`) and pick the matching row
out of stdout, rather than blindly taking the first.

### 2. CWD must be the bench's directory (fixed in audit Tier 8)

`bench.cu` files use `cuModuleLoad("hgemm.sm_86.cubin")` with a
relative filename, so they require the bench to run from its own
directory. `run_benchmark()` previously called `system2(exe_path)`
from whatever CWD R had, hitting `cuModuleLoad` failures.

Fixed in this commit:

```r
abs_exe <- normalizePath(exe_path, mustWork = TRUE)
exe_dir <- dirname(abs_exe)
prev_wd <- getwd(); setwd(exe_dir); on.exit(setwd(prev_wd), add = TRUE)
out <- system2(abs_exe, args, ...)
```

### 3. Stale `.py` references in historical issue records

`.github/issues/{08,15}` reference `scripts/bench_regress.py`. That
file was retired ages ago in favour of the R port (CLAUDE.md tooling
policy: R-only). Issue records are intentionally left as historical
records (audit Tier 4 policy), so this is informational only.

`.githooks/pre-push` was updated in commit `540adc7` to invoke
`scripts/bench/bench_regress.R`.

## Acceptance criteria

- [x] `bench_regress.R` parses the correct kernel row for every entry
      in `docs/baselines.json` (baselines schema extended with `match`
      / `section` / `value_label` per-config and `exe` per-kernel).
- [x] All 7 baseline kernels run with `Rscript scripts/bench/bench_regress.R`
      and report sensible numbers (6/7 within ±5% of `baselines.json`;
      1 real CUDA-13.2 regression on `igemm_sparse_tiled` 2048³ —
      tracked separately as a re-baseline action).
- [ ] Pre-push hook can run without `--no-verify` on a clean tree
      (blocked by the residual igemm_sparse_tiled regression).
- [x] Regression tests for the parser added at
      `tests/bench_regress/test_parser.R` (14 test groups, 32
      assertions covering hgemm multi-kernel rows, sparse multi-column
      lines via `value_label`, conv2d section bracketing, TOPS unit
      handling, edge cases).

## Bypass for now

```bash
git push --no-verify
```

## Related

- `.githooks/pre-push`           — invokes the script
- `scripts/bench/bench_regress.R` — the script in question
- `docs/baselines.json`          — recorded numbers per kernel
- `.github/issues/08-performance-regression.md` — original spec

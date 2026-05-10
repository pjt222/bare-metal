---
title: "scripts/bench/bench_regress.R: parser matches wrong kernel row, working-dir bug, .py refs"
labels: ["bug", "tooling"]
---

## Problem

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
"good" variant later. Example, `phase2/hgemm/bench`:

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

- [ ] `bench_regress.R` parses the correct kernel row for every entry
      in `docs/baselines.json` (extend baselines schema with a row
      label / regex).
- [ ] All 7 baseline kernels run with `Rscript scripts/bench/bench_regress.R`
      and report numbers within ±5% of `baselines.json`.
- [ ] Pre-push hook can run without `--no-verify` on a clean tree.
- [ ] Add a regression test for the parser (`tests/bench_regress/`?)
      that feeds canned bench stdout and asserts the right row is picked.

## Bypass for now

```bash
git push --no-verify
```

## Related

- `.githooks/pre-push`           — invokes the script
- `scripts/bench/bench_regress.R` — the script in question
- `docs/baselines.json`          — recorded numbers per kernel
- `.github/issues/08-performance-regression.md` — original spec

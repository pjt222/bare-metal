# Session handoff

> Last updated: 2026-06-03 (#134 CLOSED — PR-A #136 `ea0e7e6` + PR-B #137 `4ed0e55` both MERGED; cuasmR 0.2.0 on main, R CMD check 0/0/1 canonical Linux R. Adversarial review: 0 blockers/0 majors. Follow-up #138 CLOSED — merged via PR #139 `f4160cb` (bench_flash_all rewired onto cuasmR + revived; imma s02/s04 left distinct). Docs-housekeeping audit pass done. Next = #135 (grid-sweep Ctrl+C re-test) / #124 (bench-all). Use WSL Linux R for R work, not Windows Rscript.exe) | Branch: main

Per-author scratchpad for picking up where the previous working
session left off. Expected to churn between sessions. Durable
documentation lives elsewhere; this file records only what is true
*now* and what is the next concrete step.

- Headline performance and per-kernel measurements:
  [`inventory.md`](inventory.md), [`comparison_to_sota.md`](comparison_to_sota.md),
  [`../README.md`](../README.md).
- Optimization postmortems: [`gpu_reflections.md`](gpu_reflections.md)
  (observation catalogue, lettered A–LL plus numbered Insights).
- Structural changes and audit history: [`../CHANGELOG.md`](../CHANGELOG.md).
- Documentation map: [`index.md`](index.md).
- Open issues: [GitHub](https://github.com/pjt222/bare-metal/issues).
- Published kernel corpus: [HF dataset `pjt222/ga104-cuda-kernels`](https://huggingface.co/datasets/pjt222/ga104-cuda-kernels)
  — re-sync with `make publish-hf` (needs `HF_TOKEN` in `.env`).

## Current state

All five development phases are complete. The active optimization
queue is near-empty; the remaining items are research-grade scope.

### Pinned headline numbers (RTX 3070 Ti Laptop, sm_86)

Post-warmup, 3-run mean unless noted. Sources in
[`inventory.md`](inventory.md).

| Target                                          | Before                  | After (canonical)                       | Speedup |
|-------------------------------------------------|-------------------------|-----------------------------------------|---------|
| Flash Attention seq=1024 b=8 h=8                | 7,154 GFLOPS (regpv)    | **11,453 GFLOPS** (v2_pipeline)         | 1.60×   |
| Flash Attention smem-only path                  | 7,154 (regpv)           | **9,998** (v2 nosmem)                   | 1.40×   |
| Cross-attention typical (1024 × 256)            | 4,036 GFLOPS            | **6,550 GFLOPS**                        | 1.62×   |
| Cross-attention CLIP-77 (256 × 77)              | 656 GFLOPS              | 530 GFLOPS                              | 0.81× (regime loss) |
| ResBlock SD UNet (N=1, C=320, H=W=32)           | 13.07 ms (289 GFLOPS)   | **1.86 ms (2,025 GFLOPS)**              | 7.01×   |
| Sparse INT8 IGEMM 4096³                         | 15,030 TOPS dense       | **35,509 dense-equiv TOPS**             | 1.39×   |
| HGEMM 16-warp (post +8 padding)                 | —                       | **31,910 GFLOPS** at 2048³ / 4096³      | —       |
| Sparse HGEMM 2:4                                | —                       | **41,721 dense-equiv GFLOPS** at 2048³  | —       |

Flash Attention plateau: ~11.5 TFLOPS = 6.6 % of FP16 Tensor Core
peak (174 TFLOPS). Path:
`regpv → lean state → Q reg cache → smem_work elimination (1.40×)
→ cp.async at 8 warps (additional +0.20×)`.

### Open GitHub issues

All optimization and build-correctness work is shipped. Open issues
are benchmark-pipeline hardening — no queued kernel work.

| #   | Title                                                          |
|-----|----------------------------------------------------------------|
| 124 | `bench-all` one-click full-corpus benchmark runner (epic)      |
| 128 | Overclocked single-kernel showcase mode (deferred)             |
| 135 | Multi-kernel × clock grid sweep tool (filed 2026-05-27)        |

Resolved 2026-06-03: **#138** via PR #139 (`f4160cb`) — bench_flash_all.R
rewired onto `cuasmR::run_bench` + `parse_throughput` and revived (dead
`phase3/` path → `kernels/attention/flash_attention`); bench_imma_s02/s04
grep-extract shape left intentionally distinct.

Resolved 2026-05-27: **#131** (lock-aware bench_regress — Phase 1
end-to-end verified, `ecae5b7` + `7bfc307` pushed) and **#125**
(clock-lock IS available host-side; closed with verification log
reference). Resolved 2026-05-22: #126/#132/#133. #127/#129/#130
resolved earlier.

Design basis: [`benchmark_methodology.md`](benchmark_methodology.md).

## Latest session — issue-queue drain + benchmark planning (2026-05-21)

Goal: resolve every open GitHub issue. All four closed. A follow-up
planning pass on benchmark-pipeline reproducibility then filed five
roadmap issues (#124-128, see "Open GitHub issues" above) and shipped
their design basis — `docs/benchmark_methodology.md` and
`scripts/probe/probe_gpu_power.R` + `probe_clock_lock.R` (commit
`a149859`). Key finding: the GPU is pinned at its 150 W VBIOS power
ceiling — no headroom; reproducible numbers need clock-locking or
cooldown, not more power.

| #   | Commit  | Resolution                                                       |
|-----|---------|------------------------------------------------------------------|
| 122 | df1b4a1 | `make clean` scoped to untracked artifacts — every deletion candidate filtered through `git ls-files --error-unmatch`, so tracked handtuned `.cuasm` / experiment cubins/sass survive. |
| 103 | cd4f34a | Cross-attention regime dispatch: `dispatch.h` (`cross_attn_pick` returns a `{cubin, symbol, smem}` descriptor — the variants are driver-API cubin kernels, not host wrappers), `bench_dispatch.cu` (9 checks pass), README section. |
| 104 | —       | Closed not-planned: fragment-shfl tracking issue, no target kernel — every TC kernel with a per-row reduction already applies the pattern. |
| 105 | —       | Closed not-planned: #96 sub-task C, speculative non-TC SASS hand-tunes, no measurement-backed target. |

Build-graph gap found and fixed while pushing the above:

| Commit  | Fix                                                              |
|---------|------------------------------------------------------------------|
| 82e726b | `.gitignore` pruned (stale `phase*/`, dead `tools/CuAssembler/`); deleted spent `scripts/fix_cuda_context.R` codemod. |
| 8d94d5a | `make test` now depends on `cubins`. The `test` target built only bench executables; benches load cubins at runtime via `cuModuleLoad`, so post-`make clean` the smoke tests ran hollow ("No kernels found", swallowed by `|| true`) and the pre-push `bench_regress.R` reported every kernel as CRASH. Full `bench_regress.R` passes with 0 regressions once cubins are present. |

## Latest session — #127 build-graph fix (2026-05-21)

Wired `flash_attn_br16_regpv` + `conv2d_implicit_gemm` benches into
`make test` (#127). Four commits on `main`, **all unpushed — push is
blocked, see below**:

| Commit  | What                                                          |
|---------|---------------------------------------------------------------|
| 69cf1fb | `bench_br16_regpv.cu` loaded `flash_br16*.sm_86.cubin`; Makefile emits `flash_attn_br16*` — corrected. Latent bug exposed by #127. |
| 7fe0b56 | `Makefile` `REGRESS_BENCH` group (2 baselined exes only); `make test` depends on it. `Closes #127`. |
| 5ae60c0 | `baselines.json` flash_attn `tolerance: 0.30` + note. The apparent 78% regression is a clock artifact — laptop GA104 caps at 1410 MHz (43 W, 50 °C, no throttle), never hits 1785 boost; `5600/7152 ≈ 1410/1785`. |
| (this file) | Handoff update. |

#127 itself is **done**: flash_attn + conv2d both report OK/regression
(not SKIP) on a clean `make clean && make test` cold build — flash_attn
OK ~76-80% within the 0.30 band, conv2d OK ~90-93%.

Post-reboot power probe: still **150 W max, no headroom** — dGPU mode
did not raise the ceiling, clock-lock plan (#125) unchanged.

### Push blocked — baselines.json is corrupt

`git push` failed: the pre-push `bench_regress.R` flags 2 regressions
in kernels **#127 never touched** — `hgemm 2048³` and
`igemm_sparse 4096³`. Investigated instead of bypassing:

- **hgemm 2048³** — 5-run spread 24674-27158 GFLOPS, uniformly
  77-85 % of the 31875 baseline. Stable distribution.
- **igemm_sparse 2048³** — every run 36800-39500 GFLOPS = **116-125 %
  *above* the 31588 baseline**. Matches the entry note's "was 39674 on
  CUDA 12.8". Baseline looks recorded low.
- **igemm_sparse 4096³** — 5-run spread 15778-24412 GFLOPS, genuinely
  **bimodal (1.55× min-max)**. Clearing the worst would need
  tolerance ≥ 0.49 — that guts the regression check.

**User report (decisive):** the 2026-05-10 baseline recording hit a
**power-supply issue, fixed mid-first-kernel**. `baselines.json`'s
first kernel is hgemm — so hgemm's baseline is suspect, and
igemm 2048³ being uniformly *above* baseline fits "baseline recorded
low under bad power". Conclusion: do not widen tolerances to paper
over this — **re-baseline** hgemm + igemm. Decision: defer to next
session.

## Latest session — #130 + #125 + re-baseline protocol (2026-05-21)

`main` is **~11 commits ahead of `origin/main`, all unpushed** — push
stays blocked until the re-baseline (next steps #1). New this session:

| Commit  | What                                                          |
|---------|---------------------------------------------------------------|
| 11a8e6b | `make test` smoke loop ran each bench exe from the repo root; benches `cuModuleLoad` cubins by a cwd-relative path, so every bench loaded no kernels (swallowed by `\|\| true`). Now runs each bench from its own dir in a subshell. `Closes #130`. |
| 009fd23 | `docs/rebaseline_protocol.md` — step-by-step procedure for the [USER] re-baseline. Cross-linked from `benchmark_methodology.md`. |
| 221f50e | `probe_clock_lock.R` crashed before doing work: `attr(character(0), "status") <- 127L` is invalid R, and `nvidia-smi` is off the `sudo` secure_path. Fixed: named local, absolute nvidia-smi path, return-then-quit so `-rgc` reset is guaranteed, output tee'd to a log. |
| d7e9492 | `#125` verdict recorded in both docs — clock-lock unavailable. |
| 64d8610 | `.gitignore` probe run logs. |

#129 investigated: the `min_clock_sm` gate is **already fully
implemented** in `classify_meta` (`scripts/bench/bench_meta.R:266` —
clock below floor → SKIPPED; clock captured pre+post in
`run_benchmark`). #129 is *not* new code. The only missing piece is a
`min_clock_sm` *value* in `baselines.json`, and that value is the
recording clock minus a margin — unknowable until the re-baseline
runs. So #129 is blocked on next-step #1, exactly as the issue text
("Properly resolved by #125") says. Resolution path documented in
`rebaseline_protocol.md` §"After re-baselining".

#125 closed this session: `probe_clock_lock.R` ran (after a crash-fix
— commit message below) → `nvidia-smi -lgc` rejected by WSL2
passthrough, exit 255 "Unknown Error". Clock-locking is **not a
usable lever**. Re-baseline records at the sustained ~1410 MHz
cold-clock unconditionally; `min_clock_sm` floor pinned to ~1380.
`benchmark_methodology.md` + `rebaseline_protocol.md` updated to
drop the clock-lock branch.

## Latest session — re-baseline measured (2026-05-21)

Built `scripts/probe/rebaseline_measure.R` — a measurement driver
(modular R: `here`/`fs`/`cli`/`data.table`; per-config warmup +
N valid samples, gated on no-throttle AND SM clock ≥ 1300 MHz so
cold-decayed samples are rejected; median reported, never writes
`baselines.json`). Ran it; **3 of 4 configs measured cleanly**:

| Config (kernel)          | Old 2026-05-10 (power-fault) | New median (cold-clock) | n | Clock      |
|--------------------------|------------------------------|-------------------------|---|------------|
| `hgemm_16warp` 2048³     | 31875 GFLOPS / 0.539 ms      | **25854 GFLOPS / 0.664 ms** | 7 | 1410-1485 MHz |
| `hgemm_16warp` 4096³     | 31765 GFLOPS / 4.327 ms      | **29969 GFLOPS / 4.586 ms** | 7 | 1410-1785 MHz |
| `igemm_sparse_tiled` 2048³ | 31588 dq-GFLOPS / 0.544 ms | **37333 dq-GFLOPS / 0.46 ms** | 7 | 1410 MHz |
| `igemm_sparse_tiled` 4096³ | 30889 dq-GFLOPS / 4.449 ms | **un-measurable — 60/60 throttled** | 0 | — |

Confirms the power-fault diagnosis: hgemm reads ~19-6 % *below* the
old baseline (recorded high under bad power), igemm 2048³ reads
~18 % *above* (recorded low). The numbers are real now.

**igemm 4096³ is un-measurable in a clean regime — 60/60 tries
throttled, 0 valid.** The heaviest kernel hits the 150 W `SwPowerCap`
wall *mid-run*, every run. This is not flaky: it is deterministic.
Cooldown between samples cannot fix it — the kernel itself draws
>150 W during its own iterations (`benchmark_methodology.md`:
"a single long kernel can still throttle mid-run; cooldown cannot
fix that"). Implication: igemm 4096³ has **no fair no-throttle
baseline possible on this hardware** — its `tolerance: 0.30`
band-aid is permanent, or the entry needs `valid_when` relaxed to
*allow* `SwPowerCap` for this one config (accepting the throttled
number as the only obtainable one). Decide next session.

**Uncommitted, deliberately:**
- `scripts/probe/rebaseline_measure.R` — placement decision pending
  (see Next steps #1 / cuasmR).
- `renv.lock` + `renv/activate.R` — `data.table` 1.18.4 + `here`
  1.0.2 added (snapshot used `force = TRUE`; `cuasmR` is recorded
  `Source: unknown` and trips pre-flight validation every snapshot —
  pre-existing, not introduced this session).
- `scripts/probe/rebaseline_results.rds` — saved by the script on a
  clean finish; holds the full per-sample data.tables.

## Latest session — proper baseline under stable power + clock-lock (2026-05-22)

Stable AC power restored. Re-ran `rebaseline_measure.R` — all 4
configs measured at boost/native clock (the 2026-05-21 cold-clock run
was itself done under unstable power). Then discovered clock-lock IS
available and used it to crack the "un-measurable" igemm 4096³.

**Final baseline set (patched into `data/baselines.json`):**

| Config (kernel)            | Baseline                      | Clock regime        | Old 2026-05-10 |
|----------------------------|-------------------------------|---------------------|----------------|
| `hgemm_16warp` 2048³       | 31789 GFLOPS / 0.540 ms       | native boost 1740-1770 | 31875       |
| `hgemm_16warp` 4096³       | 30397 GFLOPS / 4.521 ms       | native boost 1770-1785 | 31765       |
| `igemm_sparse_tiled` 2048³ | 36892 dq-GFLOPS / 0.466 ms    | native 1410         | 31588          |
| `igemm_sparse_tiled` 4096³ | **50497 dq-GFLOPS / 2.722 ms** | **locked 1605 MHz** | 30889          |

**Clock-lock — #125 verdict was wrong, #125 reopened.** #125 closed
"unavailable" on a WSL-side test only. The **Windows-host
`nvidia-smi.exe -lgc MIN,MAX`** works (driver 595.97, elevated shell);
a host-side lock applies to the whole GPU incl. WSL CUDA. `-rgc`
restores boost.

**igemm 4096³ was never un-measurable — it was throttle-contaminated.**
The bench averages 50 launches; `SwPowerCap` hits a varying fraction
mid-run → ~1.9× bimodal average. Locked at 1605 MHz it is stable
(spread 1.02×, 11/11 within ±5%). Sweep (`scripts/probe/clock_lock_sweep.R`,
results in `clock_lock_sweep.rds`): 1200→37k, 1410→44k, 1500→47k,
1605→50.5k, 1710→50.9k, 1785→52k-but-SwPowerCap-active. Above ~1605
the kernel is on a **power-bound plateau** — more clock just sags back
under the cap. 1605 is the highest clock the lock holds cleanly.

**igemm 4096³ removed from the automated gate.** A locked-1605 value
can't be regression-checked by `bench_regress.R` (re-measures at
native boost → false alarms). Recorded as a documented reference in
`docs/inventory.md`; gating it needs a lock-aware harness — **filed
as #131**.

Pipeline fix: `clock_lock_sweep.R` warmup 8→20 (8 didn't settle the
GPU at high clock — recurring slow first sample was a warmup
artifact, confirmed gone at 20).

All baseline + clock-lock work committed and pushed (`main` synced).

## Latest session — issue-queue drain (2026-05-22)

Resolved the cleanly-resolvable open issues; `main` synced with origin.

| #   | Resolution                                                       |
|-----|------------------------------------------------------------------|
| 132 | docs CI Quarto render was red since 2026-05-21. Two causes: workflow omitted `knitr`/`rmarkdown` (`04514da`); then source `fs` failed to compile under renv. Fixed `25884aa` — disable renv autoloader for the job + `use-public-rspm` binary packages. First diagnosis (cuasmR aborting `renv::restore`) was wrong — the workflow never calls `restore`. |
| 126 | GPU-mode metadata — `capture_gpu_state()` reads `BARE_METAL_GPU_MODE` env var → `$host$gpu_mode`; `unknown` if unset. `d21cdd0`. |
| 133 | cuasmR renv hygiene — reinstalled via `renv::install("./R/cuasmR")`, recorded `Source: Local` (was `unknown`); `renv::snapshot()` no longer needs `force`; `renv::status()` clean. `588c773`/`c364f49`. |

Build-graph gap found + fixed: `.gitignore` `bench`/`bench_*` rules
swallowed the `scripts/bench/` R-source dir — `bench_meta.R`,
`bench_reference.R`, `compare_reference.R` were **untracked** (fresh
clone would break the pre-push hook + Makefile). `832201d` fixes the
rule; `d21cdd0`/`66e53b9` track the three files.

`#125` reopened — the 2026-05-21 close ("clock-lock unavailable") was
wrong; it works host-side. New issue `#134` filed for the full
CRAN-ready cuasmR migration (probe/bench scripts → package functions).

## Latest session — #131 Phase 1, lock-aware bench_regress (2026-05-22)

Implemented Phase 1 of the #131 plan (the plan that lived in this file
last session — now executed, section removed). One commit, **`ecae5b7`,
local and UNPUSHED**:

| File                          | Change                                          |
|-------------------------------|-------------------------------------------------|
| `data/baselines.json`         | New optional per-config `clock_lock` field (int MHz), documented in the `schema` block. Re-added the `igemm_sparse_tiled` `4096_4096_4096` entry: `clock_lock: 1605`, 50497 dq-GFLOPS / 2.722 ms. |
| `scripts/bench/bench_regress.R` | `--clock-locked <MHz>` CLI flag; `measure_clock_locked()` helper (warmup 20, median of 5 valid runs, two-sided clock-band check `clock_lock ± 30 MHz`); per-config dispatch in the main loop; `--list` annotation. |
| `docs/benchmark_methodology.md` | Rewrote the stale "clock locking NOT available" section — host-side `nvidia-smi.exe -lgc` works. Documents the `--clock-locked` workflow. |

**Behaviour.** A `clock_lock` config is SKIPPED unless `bench_regress.R`
is run with `--clock-locked <MHz>` matching the entry. When it matches,
the kernel is measured as a median of N valid runs; any sample whose
observed SM clock leaves the band is rejected. Band check is two-sided
— a clock far *above* `clock_lock` means the lock was never applied →
`INSUFFICIENT`, not a bogus pass. The pre-push hook runs without the
flag, so igemm 4096³ SKIPs there by design.

**Verified (3 of 4 paths, all from WSL):**

| Path                                                  | Result |
|-------------------------------------------------------|--------|
| No flag → igemm 4096³ SKIPPED                         | ✅ PASSED |
| Mismatched `--clock-locked 1500` → SKIPPED            | ✅ |
| `--clock-locked 1605`, GPU **not** locked → band rejects all 20 attempts (SwPowerCap + clock 1410 outside 1575-1635) → INSUFFICIENT, no false pass | ✅ |
| Full unfiltered sweep (pre-push equivalent) → 7 configs, 0 regressions, PASSED | ✅ |
| **Real host-side lock 1605 → measures ~50497**        | ⏳ **unrun** — needs an elevated Windows shell |

The 4th path is the only gap. `ecae5b7`'s message says `Closes #131,
#125`; do not push until that path is verified, or amend the message
first. See Next steps #1.

## Latest session — #131 verify + close, grid-sweep tool filed (2026-05-27)

Closed the #131 verification gap and packaged the elevated workflow as
a reusable script. `main` synced with origin (3 commits pushed:
`ecae5b7`, `6da2a20`, `7bfc307`).

| Commit  | What                                                          |
|---------|---------------------------------------------------------------|
| 7bfc307 | `scripts/probe/run_locked_eval.ps1` — elevated-pwsh driver. Asserts Administrator, locks SM clock via `nvidia-smi.exe -lgc`, calls `wsl.exe ... Rscript bench_regress.R --clock-locked`, restores via `-rgc` in `finally`. Captures pre/during/post GPU snapshots; logs full bench stdout to `scripts/probe/eval_logs/<stamp>_<kernel>_<mhz>.log` + structured JSONL row to `eval_logs/results.jsonl`. PS 5.1 + PS 7 compatible. `.gitignore`: `*.rds` + `eval_logs/`. |

**Path 4 verified end-to-end** (the 2026-05-22 gap):

| Path                                                            | Result |
|-----------------------------------------------------------------|--------|
| Real host-side lock 1605 (`nvidia-smi.exe -lgc 1605,1605`) → igemm 4096³ measures **50297 dq-GFLOPS = 99.6% of 50497 baseline, OK** | ✅ |
| Pre-push hook: 7 configs, 0 regressions, igemm 4096³ SKIPPED as designed | ✅ |
| `-rgc` restored boost (P0, 1770 MHz, 48 W)                       | ✅ |

#131 auto-closed by `ecae5b7` trailer. #125 was NOT auto-closed (the
`Closes #131, #125` form with a comma trips GitHub's parser — the
second issue needs its own `Closes` keyword); closed manually with a
reference to the verification log. Lesson: use one `Closes` per issue
in trailers.

**#135 filed** — multi-kernel × clock grid sweep tool. Deliberately
scoped as a separate orchestrator (not an extension of
`run_locked_eval.ps1`), with declarative YAML spec, pure R inner
measurement function (destined for cuasmR per #134), structured rds
output, Ctrl+C-safe `-rgc` via `[Console]::CancelKeyPress`,
`-Resume` after crash, and `-DryRun` validation. Full design in the
issue body — implement in a dedicated session.

## Latest session — #135 grid sweep tool, Phases 1+2+4+5 (2026-05-27)

Implemented the grid-sweep tool end-to-end in 5 commits. `main`
synced; pre-push hook PASSED each time (7 configs, 0 regressions).

| Commit  | What                                                          |
|---------|---------------------------------------------------------------|
| 7bd55b4 | Phase 1 — Ctrl+C-safe orchestrator skeleton. Lock-state sentinel at `eval_logs/.LOCK_HELD`; C# `Add-Type` cancel handler (PS script-block handlers crash with "no Runspace available" — the .NET cancel event fires on a worker thread with no PS Runspace, T2 in Phase 1 testing proved this). Dummy `Start-Sleep` inner. Manual T1, T2, T2-clean, T4, T4b all PASSED. |
| 02b2831 | Phase 2 — YAML spec + R planner/measurer + full orchestrator. `scripts/probe/grid_sweep.yml` declarative spec (6 kernel × N regimes = 28 cells). `scripts/probe/grid_measure.R` with `--mode plan` (writes cell list to `--out` JSON file; renv NOTEs on stdout would otherwise pollute pipe-capture) and `--mode measure` (per-cell, two-sided clock-band check, JSONL append per sample). `run_grid_sweep.ps1` consumes plan, groups by clock, locks per group, restores between groups. C# ChildPid tracking + explicit `wsl.exe -e pkill -f grid_measure.R` because `Process.Kill(tree=true)` only walks Windows tree — WSL2 Linux processes survive otherwise. `-OnlyCellId`, `-DryRun`, `-Resume`, `-NoLock`, `-ForceClearSentinel` switches. |
| 8903337 | Phase 4 + 5 — `scripts/probe/grid_collect.R` materialiser (tolerant JSONL→RDS reader, `--print` summary with per-(cell,clock) median + reject-reason histogram). `docs/grid_sweep_methodology.md` documents architecture rationale (JSONL-not-RDS for atomicity; C# handler for Runspace problem; WSL pkill for tree-kill gap), canonical resume key `(git_head, clock_target_mhz, cell_id)`, regime-selection guidance, failure-mode → diagnostic table, six invariants. |
| 4865710 | Fix from P2-5 elevated test — user reported "skip, not stop". Ctrl+C reached the foreground child (wsl→Rscript) before pwsh's C# CancelKeyPress handler. R died with exit 1; orchestrator's per-cell try/catch treated as recoverable, applied NEXT group's lock, continued. Two-sided fix: R traps `interrupt` → exits 130 distinct from generic 1; PS sweep loop checks `cellExit == 130` and breaks both loops via `:groupLoop` label. |
| 246c961 | Fix from P2-5 re-test — user reported "works, but needed three Ctrl+C". Each press only killed the current sample's bench child (which exits 130 = 128 + SIGINT); R didn't see the signal itself (consumed by child) and proceeded to next sample. Fix: per-sample loop checks `r$rc == 130L` immediately after `run_one_sample` and propagates as cancel. Single press should now suffice. **Re-test pending in next session.** |

#135 acceptance criteria status:
- [x] `grid_sweep.yml` with the 6 baselined kernels
- [x] `grid_measure.R` standalone, unit-smoke (smoke-002) passes headless
- [x] `run_grid_sweep.ps1` runs the full spec under elevated pwsh
- [x] Resume verified (Step 1: 28 cells measured 3 → Step 2: 25 pending)
- [~] Ctrl+C verified — partially. C# handler PASSED Phase 1 T2; mid-measure abort verified at exit 130 path, single-press fix unverified.
- [x] `docs/grid_sweep_methodology.md` written
- [x] `run_locked_eval.ps1` UNCHANGED (separation preserved)

Field-validated cell measurements from the partial P2-5 run
(igemm_sparse_4096 plateau, elevated 1605 NOT in this run because
that's the cancelled cell — but 1200/1410/1500 measured cleanly):
- 1200 MHz: median 36293 dq-GFLOPS (spread 31673-37876)
- 1410 MHz: median 42108 dq-GFLOPS (spread 41584-45771)
- 1500 MHz: median 47019 dq-GFLOPS (spread 32414-48595 — one sample dropped to 32k, possible mid-lock blip)

These match the clock_lock_sweep.R numbers (1410→44k, 1500→47k)
within run-to-run variance. Sanity-check passes.

## Latest session — #134 PR-A COMPLETE (Phases 0–5) (2026-06-02)

The #134 cuasmR measurement-migration runs as **two PRs** (decided with
user): PR-A = the dedupe (Phases 1–5), PR-B = CRAN polish (Phase 6 —
roxygen/`man/` + `R CMD check --as-cran`). **PR-A is feature-complete**
on branch **`feat/134-cuasmr-measurement-migration`** (~10 commits on
`53d7671`). **Push initiated** end of session (user approved push + PR);
the pre-push hook runs `make test` (builds + smoke-runs all benches —
several minutes of GPU work) + `check_links.R` + `bench_regress`, so the
push is slow. **First confirm the branch actually landed on origin**
(`git ls-remote --heads origin feat/134-cuasmr-measurement-migration`):
if yes → `gh pr create` (PR not yet opened); if not → re-push (the hook
may have been interrupted; the gate itself passes — verified). Each phase
kept the pre-push gate green; incremental, never big-bang.

Plan file: `~/.claude/plans/hey-there-how-tender-fern.md`.

| Phase | Commit | What |
|-------|--------|------|
| 0 | — | Baseline gate: 7 configs, 0 regressions, 5 OK + 2 SKIP, PASSED. Reference for diffing. |
| 1 | `e01cf4d` | `bench_meta.R` → cuasmR (`capture_gpu_state`/`classify_meta`/`decode_throttle`/`summarise_meta`). Source-time WSL `LD_LIBRARY_PATH` guard → `.onLoad` (`zzz.R`). 4 sourcers `source()`→`library(cuasmR)`; compat shim kept. **`.gitignore` fix**: `bench_*` rule was swallowing `R/cuasmR/R/bench_*.R` — added `!R/cuasmR/R/bench_*.R` (cf. `832201d`). |
| 2a | `c9f977d` | `run_bench(exe, args, timeout=0)` → `bench_run.R`. 4 runners consolidated. |
| 2b | `caeca97` | `parse_throughput(lines, match, section, value_label, pick)` → `bench_run.R`. Unifies 4 divergent parsers; **GPU-free differential test** (`test-parse_throughput.R`, 41 assertions) with the 3 originals as oracles over real captured stdout — incl. igemm_sparse two-number line (dense-equiv not eff) + conv2d section. |
| 3 | `d518f1b` | `validate_sample` + `collect_valid_samples` + `report_median_metrics` → `bench_measure.R`. Independent re-derivation (workflow, one agent/caller) caught **two drifts the planning agents missed**: (a) grid validates throttle on POST only — pass `(post, post)` to collapse `classify_meta`'s pre/post union; (b) `classify_meta` min_clock_sm did `NA < x` (`if(NA)` crash) — made NA-safe (reject). `test-validate_sample.R` (58 assertions) asserts each caller's verbatim decision incl. the discriminating pre-throttled/post-clean case + NA clock. grid keeps its record-all loop. |
| 4 | `fd8f853` | `append_jsonl_row` + `read_jsonl` → `bench_io.R`. The tolerant per-line read was dup'd in grid_measure's `load_jsonl_keys` and grid_collect's `read_jsonl`; both rewired. `test-bench_io.R` (12). |
| 5 | `65855a0` | `check_regression` → `bench_regression.R` (last API stage). bench_regress drops local. Also deduped probe_gpu_power's identical `decode_throttle`. `test-check_regression.R` (15). |
| bump | `72da9b2` | cuasmR `0.1.0`→`0.2.0` (new API surface); renv.lock Version field synced (surgical edit, not a full snapshot). |

NAMESPACE is **hand-maintained** in PR-A (roxygen `#'` blocks present but
`roxygenise()` deferred to PR-B/Phase 6 — running it now churns the whole
NAMESPACE/`man/`).

Verification each phase: reinstall (`Rscript scripts/install_cuasmR.R` —
**mandatory**, `library()` loads the *installed* copy) → run rewired
script(s) → gate green. Live native grid cell measure confirmed
run_bench+parse_throughput+JSONL end-to-end (hgemm_2048 → 31889 GFLOPS).

### PR-A verification (all green)

- **Full cuasmR suite: 131 assertions, 0 fail** (parse_throughput 41,
  validate_sample 58, bench_io 12, check_regression 15, roundtrip 6 — the
  1 warning is the pre-existing roundtrip one). All GPU-free differential/
  unit tests; they become the PR-B/Phase-6 test base.
- **Gate unchanged at every phase**: 7 configs, 0 regressions, PASSED.
  `--tolerance 0.0` → REGRESSION detected, exit 1 (the gate still *blocks*
  through `cuasmR::check_regression`).
- **Live integration**: grid native cell (validate_sample(post,post) +
  record-all JSONL); bench_regress `--clock-locked 1605` with NO host lock
  → igemm_4096 INSUFFICIENT (0/5 valid, all rejected, no false pass);
  rebaseline config 1 → clean median `31888.7 GFLOPS (31637.2-32020)` via
  the rewired closure + `on_sample` glue + `report_median_metrics`;
  probe_gpu_power `--json` decodes `0x..01`→GpuIdle via cuasmR.
- **Downstream safe**: `bench_reference.R` `source()`s bench_regress.R; its
  `sys.nframe()==0L` guard means no gate auto-run, and `check_regression`
  (cuasmR) / `run_benchmark` (kept) / `capture_gpu_state` (cuasmR) all
  resolve. No deleted `.parse_line`/`.pick_line` usage anywhere.

**AC4 status**: every extracted measurement function is deduped. The one
loose runner left is `bench_flash_all.R`'s `run_bench` — deliberately NOT
migrated (flash-specific run+parse fusion: GFLOPS + PASS/FAIL, no GPU-state
capture; behaviour + flash-bench verification make it a separate follow-up).

The packaged measurement API (cuasmR 0.2.0), in pipeline order:
`run_bench → parse_throughput → validate_sample → collect_valid_samples →
report_median_metrics → check_regression`, plus `append_jsonl_row` /
`read_jsonl` and the `capture_gpu_state` / `classify_meta` /
`decode_throttle` / `summarise_meta` base. #124 (`bench-all`) builds on this.

## Next steps

1. ✅ **DONE + MERGED (2026-06-03) — PR-A [#136](https://github.com/pjt222/bare-metal/pull/136) merged to main, merge-commit `ea0e7e6`**
   against `main`. Pre-push gate green (7 configs, 0 regressions, 1
   improvement conv2d 113%, 2 skip). PR body = phase table + verification +
   bench_flash_all defer; **references #134 without closing** (no
   `close/fix/resolve #134` adjacency — GitHub ignores negation, so even
   "does not close #134" would auto-close; phrased as "leaves #134 open").
   PR-B/Phase 6 still owes CRAN.
   - **Hang gotcha (root cause of last session's "push didn't land"):** an
     *interrupted* push leaves the smoke bench (`./bench 512 512 512`)
     spinning ~92% CPU, GPU 0% util, `wchan=0` — it wedges the WSL CUDA
     path so the *next* push hangs on the **first** smoke bench (looks slow,
     is stuck; 20 min, no log progress). Fix: `kill -9` the spinner tree,
     confirm a fresh `./bench 512 512 512` runs clean, re-push. No
     `--no-verify` / `wsl --shutdown` needed — the kill is the root fix.
2. ✅ **DONE + MERGED (2026-06-03) — PR-B [#137](https://github.com/pjt222/bare-metal/pull/137) merged to main, merge-commit `4ed0e55`**
   (was stacked; retargeted to `main` after #136 merged, diff stayed 707/58/31
   = no double-count). GitHub auto-retargeted to
   `main` when #136 merged. `closingIssuesReferences == []` (verified — does
   NOT auto-close #134; leaves the epic open). Pre-push gate green.
   - **`R CMD check --as-cran` (canonical Linux R 4.6.0): 0 errors / 0 warnings
     / 1 note** (CRAN incoming feasibility: new submission + `sm_8x` title-case
     false positive). The Windows R 4.5.2 run showed 0/0/**3** — the 2 extra
     NOTES were just missing-pandoc + clock-skew (Windows env), not defects.
     `cran-comments.md` documents the conservative 3-NOTE list (harmless;
     CRAN's Linux sees 1). Code-level result (0E/0W) holds on both.
   - Did: `roxygenise()` → `man/` (20 `.Rd`) + roxygen-owned NAMESPACE (export
     set unchanged, 20); `DESCRIPTION` `Imports: jsonlite, stats, utils` +
     Title `cubins`→`Cubins`; `NEWS.md`; `.Rbuildignore`; `cran-comments.md`.
     Source fixes the check surfaced: invalid `attr(character(0),"status")<-1L`
     in `run_bench` (→ `structure(...status=1L)`). NB the old form did NOT
     silently read rc=0 — `attr(<literal>,k)<-v` *throws* "target of assignment
     expands to non-language object"; inside the tryCatch error handler that
     propagates and aborts the whole harness. The fix converts that propagating
     crash on the can't-launch path into a clean rc=1 (→ `validate_sample`
     records `crash(exit=1)`). Also `utils::modifyList`, non-ASCII (em-dash→`--`,
     °→`°`).
   - testthat 131 pass / 0 fail / 1 warn (independently re-run post-merge).
     bench_flash_all `run_bench` dup + bench_imma_s02/s04 `run_bench_grep` +
     the latent crash twin now tracked in **#138** (comparison-harness
     consolidation, low-pri).
   - **Toolchain (CORRECTED):** I first ran roxygenise + check via the
     **Windows** `Rscript.exe` (anchored on the global CLAUDE.md MCP path) —
     a mistake. The gate, Makefile, and hook all use the **WSL Linux R**
     (`/usr/local/bin/Rscript` 4.6.0) with `renv/library/linux-ubuntu-noble/`.
     Use Linux R for R work here (it has roxygen2; for the check use base
     `R CMD check` + `_R_CHECK_FORCE_SUGGESTS_=false`). See memory
     `project_r_toolchain_renv_gotchas`.
   - **renv lib — NO real problem (earlier note was wrong).** The "degraded to
     3 packages" lib was the **Windows** `renv/library/windows/` lib, which
     the gate never uses. The **Linux gate lib** (`linux-ubuntu-noble`) is
     healthy: **72 packages** incl. cuasmR 0.2.0, jsonlite, stringr — which is
     why the push hook (Linux R) passed. No restore needed. renv.lock fine.
3. **[USER] Re-test P2-5 with `246c961`** (separate from #134). In elevated pwsh:
   `pwsh -File D:\dev\p\bare-metal\scripts\probe\run_grid_sweep.ps1 -OnlyCellId igemm_sparse_4096`.
   Wait for first sample line of any group, press Ctrl+C **once**.
   Expect: `Bench exited 130 (SIGINT)` → `Cell cancelled by user`
   → cleanup → exit. Verify no orphans:
   `Get-Process Rscript -ErrorAction SilentlyContinue; wsl -- pgrep -f grid_measure.R`.
   If single press still requires multiple, investigate further.
4. **P2-6 — full elevated sweep (~1 h).** Once P2-5 re-test is green,
   run the full plan:
   `pwsh -File D:\dev\p\bare-metal\scripts\probe\run_grid_sweep.ps1`.
   Then materialise: `wsl -- Rscript scripts/probe/grid_collect.R --print`.
   Inspect `grid_sweep_results.rds` for the full plateau map. Close #135.
5. **#124 — `bench-all` runner** (epic). Build on the packaged
   cuasmR API (#134) once it exists.
6. **#128 — OC showcase**: deferred. The grid_sweep above-native-
   clock data will be its data source.

Closed earlier, not in scope unless reopened:

- **#32 polyhedral spring networks** — literature scoping lives in
  `docs/polyhedral_spring_networks.md`. Re-open only if a kernel
  implementation is wanted.

## Hardware constraint (recap)

GA104 sm_86, RTX 3070 Ti Laptop. 48 SMs (desktop bin) or 46 SMs
(laptop bin). 100 KB max shared memory per SM. FP32 21.7 TFLOPS,
FP16 Tensor Core 174 TFLOPS, INT8 Tensor Core 348 TOPS, DRAM
608 GB/s, L2 4 MB.

The 50 KB shared-memory cliff is the dominant tile-size constraint:
blocks at ≤50 KB run at 2 blocks/SM (8 warps, latency hidden); blocks
at >50 KB drop to 1 block/SM (4 warps, exposed DRAM stalls,
measured 2× regression).

The four laws of GA104 (full statement in `../AGENTS.md` and
`docs/tutorial/06-the-four-laws.md`):

1. Feed Tensor Cores continuously.
2. Read each byte of DRAM at most once per kernel.
3. Fill the warp schedulers — 8 warps/SM is the floor.
4. Never cross the 50 KB shared-memory cliff.

## Prior session log

Removed from this file. The 677-line per-session walls that
previously lived here are preserved in git history; relevant
durable findings have been integrated into `gpu_reflections.md`
(observations) and `CHANGELOG.md` (structural changes). To recover
a specific prior session, browse:

```bash
git log -- docs/CONTINUE_HERE.md
git show <commit>:docs/CONTINUE_HERE.md
```

Notable prior session anchors, in chronological order:

| Date       | Anchor commit | Summary                                                      |
|------------|---------------|--------------------------------------------------------------|
| 2026-05-07 | (sprint plan) | Filed #84–#96 sprint plan; most items now closed by Obs U–LL. |
| 2026-05-08 | f2100ac       | NCU profiling session; 7 issues closed; +8 padding pattern. |
| 2026-05-10 | b18dc1b       | Kernel tree reorganized from `phaseN/` to family directories. |
| 2026-05-10 | 6cf4161       | Dark-theme ggplot pass; figures re-rendered.                 |
| 2026-05-12 | 82bc175       | Sprint-queue reconciliation against `gpu_reflections.md`.    |
| 2026-05-21 | ae5c69a       | Docs review + HF publish (epic #110): README→abstract, docs CI, HF dataset published; build fixes #116/#117/#119/#120/#121. |

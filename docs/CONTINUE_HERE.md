# Session handoff

> Last updated: 2026-05-27 (#131/#125 closed; #135 grid-sweep tool filed) | Branch: main

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
| 134 | Migrate probe/bench R tooling into cuasmR; make cuasmR CRAN-ready (epic) |
| 135 | Multi-kernel × clock grid sweep tool (filed 2026-05-27)        |

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

## Next steps

1. **#135 — grid sweep tool**. Implement per the design in the issue:
   YAML spec at `scripts/probe/grid_sweep.yml`, `grid_measure.R`
   inner, `run_grid_sweep.ps1` orchestrator. Includes Ctrl+C-safe
   cleanup, lock-state sentinel, resume support, schema validation
   before lock. New session.
2. **#134 — cuasmR CRAN-ready migration** (epic). Refactor loose
   `scripts/probe/*.R` + `scripts/bench/*.R` into documented, tested
   `cuasmR` functions; `R CMD check` clean. `measure_clock_locked()`,
   the `rebaseline_measure.R` sampling loop, and (when shipped) the
   #135 `grid_measure.R` function are duplicate-by-design for now —
   dedupe into the package here.
3. **#124 — `bench-all` runner** (epic). Build on the packaged cuasmR
   API (#134) once it exists.
4. **#128 — OC showcase**: deferred. The clock-lock sweep is the
   groundwork — `-lgc` above native clock is exactly the OC lever.
   Likely consumes #135's grid_sweep output.

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

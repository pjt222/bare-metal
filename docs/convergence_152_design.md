> **Status:** Design proposal for [#152](https://github.com/pjt222/bare-metal/issues/152) — NOT yet implemented.
> Produced 2026-06-05; line citations verified against `main` @ `456e9a7`.
> Resolves the 5 open design questions on the convergence epic (unify
> `bench-all` #124 + `grid sweep` #135 into: full corpus × {native ∪ locked-clock grid}).
>
> **Provenance (negative space):** synthesized by a 16-agent design workflow —
> 3 faithful source maps (bench-all half, grid-sweep half, cuasmR API) → a
> 3-spine design panel (extend-in-place / orchestrator-led / store-first) →
> per-design judge + adversarial API-grounding check + Q2×Q4 nesting attack →
> coherence-aware synthesis. The adversarial pass earned its keep: it **cracked
> the orchestrator-led spine at its premise** by reading `grid_measure.R:100`
> (`if (is.null(k$regimes)) k$regimes <- spec$clocks` resolves an omitted
> `regimes` to the *full grid*, not `[native]`) — which became the winning
> design's single most important spec rule (the converged planner must implement
> the `[native]` default explicitly). All line citations below were re-verified
> against source by the orchestrator before this doc was written.
>
> **Reading the citations:** every `file:line` reference points at *current*
> code (HEAD `456e9a7`) — the structure being reused or extended. Proposed new
> artifacts (the regime loop, `--regime`/`--plan-regimes`/`--resume` flags,
> `bench_all_collect.R`) are described by name, never by a line number.

# #152 — Unified Benchmark Tooling: Full Corpus × {native ∪ locked-clock grid}

## Decision

Converge the two shipped halves — `bench_all` (#124, full 48-exe corpus, native-only, never-aborts) and the `grid sweep` (#135, 7-kernel locked-clock grid under an elevated pwsh lock) — into their union by **extending `bench_all.R` in place** with a regime dimension, while the **host clock lock lives strictly outside R** in a generalized `run_grid_sweep.ps1`. No third measurement path is created; the only behavioral change inside R is passing the already-existing `validate_sample(clock_band=)` argument.

The governing structural rule: **the lock lifecycle is owned entirely by pwsh (outside R); the retry-to-min-valid loop runs entirely inside the R child.** This makes the Ctrl+C-mid-retry crux safe *by construction* — no code path lets the retry loop hold the lock.

## The 5 resolved questions

### Q1 — Spec unification

ONE schema. `bench_all.yml` gains a per-kernel `regimes` field (list of int MHz or the string `native`) **defaulting to `[native]`** — the exact field/semantics already shipped in `grid_sweep.yml`. It also merges `grid_sweep.yml`'s top-level `clocks: [1200,1410,1500,1605,1710,native]` (the fallback list a kernel resolves to **only** when it explicitly writes `regimes: <clocks>`) and `defaults: {band_mhz: 30, n_samples, warmup}`.

**Critical default rule** (differs from grid's shipped behavior): a measurable kernel that *omits* `regimes` defaults to `[native]`, NOT the full clocks grid. `grid_measure.R:100` (`if (is.null(k$regimes)) k$regimes <- spec$clocks`) resolves an omitted `regimes` to the *full* grid — the converged planner must implement the `[native]` default explicitly. This preserves `bench_all`'s native-only default for the ~40 un-swept corpus exes.

The measurable / non-measurable / skipped taxonomy and the verified/infer flag are untouched (`identical(k$measurable, FALSE)` / `identical(k$run, FALSE)` / `isTRUE(k$verified)` reads in `mk()`).

**Taxonomy × regime rule (the genuinely-open sub-part):** a `measurable:false` or `run:false` bench runs **ONCE-NATIVE**, never once-per-regime. The regime loop is entered ONLY for the measurable branch; `effective_regimes = (if measurable && run) kernel$regimes %||% [native] else [native]`. Rationale: a clock lock only varies a clock-sensitive *throughput*; a bench that emits no parseable single number (A/B sweep table, ms-only pipeline, correctness harness) has nothing for a band to gate, so N regime-copies would be N identical `non-measurable`/`skipped` records that waste elevated lock time and risk leaking a `non-measurable` row into a locked-perf bucket. A declared `regimes` on a non-measurable entry is ignored + warned.

### Q2 — Lock orchestration

**Wrap `bench_all.R` under the generalized grid orchestrator** — do NOT teach pwsh the corpus (that would fork `merge_spec`/`discover_corpus`/the taxonomy into PowerShell). `run_grid_sweep.ps1` keeps its elevation assert, `.LOCK_HELD` sentinel gate, C# `GridSweepCleanup` CancelKeyPress handler, `Apply-Lock`/`Release-Lock`, and group loop **verbatim**; only the unit of work changes from "one `grid_measure.R --mode measure` per cell" to "one `bench_all.R --regime M` per clock group". A new `bench_all.R --plan-regimes --out <file>` subcommand emits the distinct regime list so pwsh knows which groups need a lock.

**Native is elevation-free two ways:** (a) `make bench-all` runs `$(RSCRIPT) scripts/bench/bench_all.R` directly in plain WSL (`Makefile:243-245`) with the default `[native]` regime and `clock_band=NULL` — zero pwsh/admin/sentinel, exactly today's behavior; (b) under the orchestrator, the native group has `needsLock = ($clockLabel -ne 'native') -and (-not $NoLock) = FALSE` (`run_grid_sweep.ps1:431`). Elevation is required ONLY when a locked regime is requested.

### Q3 — Output store

JSONL is the single source of truth; RDS is a regenerable rollup. `bench_all` already writes one `append_jsonl_row` per attempt. Extend the attempt row with `regime`, `clock_target_mhz`, `band_lo`/`band_hi`, and (grafted) the taxonomy columns `spec_source`/`measurable`/`verified`. One store per run (`results/bench_all/<ts>/samples.jsonl`) plus a cross-run rollup RDS (`results/bench_all/bench_all_results.rds`) materialized by a new `bench_all_collect.R` (`read_jsonl(simplify=TRUE)` → `rbindlist(fill=TRUE)` → `saveRDS`). `results.json`/`summary.md` stay as per-run human artifacts, each gaining a `regime` key; `summary.md` keeps its three taxonomy buckets, sub-grouping the measurable-known bucket by regime.

**Store key: `(git_head, cell_id, regime)`** — grid's exact three-tuple, with `cell_id == bench_all id == grid id` and `regime == clock_label`. `attempt` is a non-key row column (Spine C's 4-tuple `(…, sample_idx)` was refused to preserve the resume-key invariant).

Field provenance note: `clock_sm_mhz` already exists in `bench_all`'s attempt row (`bench_all.R:417`); `clock_mem_mhz` is a NEW harmonization-toward-grid addition; `throttle` is `bench_all`'s name (grid's `throttle_str`), aligned on read via `fill=TRUE`.

### Q4 — Carry-over invariants

A preservation checklist, not a fork. See the invariant table below. Headlines: never-abort survives because the regime loop is *outside* the per-config `tryCatch` (`bench_all.R:603-608`) and pwsh continues on `cellExit!=0` non-130; retry-to-min-valid reuses `collect_valid_samples` verbatim with only the `validate_fn` closure carrying `clock_band`; the two-sided band uses `validate_sample`'s existing `clock_band=c(lo,hi)`; the advisor invariant is held by once-native non-measurable runs + the three segregated `summary.md` buckets + taxonomy store columns.

### Q5 — Reuse the cuasmR API

No third measurement path. The regime loop wraps the SAME five-stage pipeline (`run_bench` → `parse_throughput` → `validate_sample` → `collect_valid_samples` → `report_median_metrics`); the only behavioral change — passing `clock_band` — uses an argument `validate_sample` already exposes (it predates #152 and is how grid's band reject works). `bench_all` keeps `collect_valid_samples` for retry-to-min-valid (grid deliberately does not use it). The pwsh wrapper invokes `bench_all.R` as a child exactly as grid invoked `grid_measure.R`. `grid_measure.R` is retired once `--regime` covers its 7 cells.

## Unified YAML (concrete)

```yaml
defaults:
  n_samples: 7        # bench_all still honors --min-valid override at runtime
  warmup: 20
  band_mhz: 30        # two-sided clock-band tolerance; locked regimes only

clocks: [1200, 1410, 1500, 1605, 1710, native]   # fallback ONLY for `regimes: <clocks>`

kernels:
  - id: igemm_sparse_4096            # MEASURABLE + power-bound: native omitted (throttles)
    exe: kernels/gemm/igemm/bench_sparse
    args: [4096, 4096, 4096]
    match: "igemm_sparse_tiled"
    value_label: "dense-equiv GFLOPS"
    verified: true
    regimes: [1200, 1410, 1500, 1605, 1710]
    note: "power-bound. Fair 50497 @ -lgc 1605."

  - id: hgemm_16warp_4096            # MEASURABLE + clock-sensitive
    exe: kernels/gemm/hgemm/bench
    args: [4096, 4096, 4096]
    match: "hgemm_16warp (128x128 2blk/SM)"
    verified: true
    regimes: [native, 1605, 1710]

  - id: conv2d_nhwc                  # MEASURABLE, regimes omitted -> [native] ONLY
    exe: kernels/convolution/conv2d/bench
    args: []
    match: "conv2d_nhwc (3x3)"
    value_label: GFLOPS

  - id: hgemm_persistent            # NON-MEASURABLE: once-native; any regimes ignored+warned
    exe: kernels/gemm/hgemm/bench_persistent
    measurable: false
    note: "A/B sweep table; no single-number line"
```

## Orchestration flow

**Division of iteration:** pwsh unrolls regimes (one child per clock group, each child a single `--regime M`); `bench_all.R` iterates configs internally within one fixed regime. There is no nested regime loop in R — the lock changes between groups host-side, so R cannot loop locked regimes internally.

**Native (lock-free, plain WSL):** `make bench-all` → `Rscript bench_all.R` directly; config loop calls `validate_sample(clock_band=NULL)`. No PowerShell, no sentinel.

**Locked (elevated pwsh):** Assert-Elevated → stale-sentinel gate → register C# CancelKeyPress → plan handshake via `--plan-regimes --out <file>` (read from file, never YAML) → group pending by clock_target (native first) → per locked group `Apply-Lock([int]M)` = `nvidia-smi.exe -lgc M,M` + `LockApplied=true` + sentinel → ONE child `bench_all.R --regime M --clock M --band <band_mhz>` measuring every measurable kernel whose regimes include M, passing `clock_band=c(M-band, M+band)` → `Release-Lock` = `nvidia-smi.exe -rgc` + clear sentinel. One `-lgc`/one `-rgc` per clock group.

**Ctrl+C-safe restore (the crux):** the retry loop is entirely inside the R child; the lock is held by the pwsh parent across the whole child. On Ctrl+C mid-retry: Route A — SIGINT reaches the R child → `on_sample`'s `if (identical(s$r$rc, 130L)) quit(status=130L)` (`bench_all.R:456`/`:478`) → child exits 130 → pwsh `cellExit==130` → `break groupLoop` → `finally Invoke-Cleanup` → `Run()` issues `-rgc` (LockApplied still true) + clears sentinel. Route B — SIGINT reaches pwsh's .NET cancel thread → C# `OnCancel` (`:243`/`:254`) runs the same idempotent `Run()` (`KillChild` = `Process.Kill(true)` + `wsl.exe -e pkill -9 -f bench_all.R`, then `-rgc`, then `Environment.Exit(130)`). The `Run()` Done-guard prevents a double `-rgc`. Belt-and-suspenders: `bench_all.R:634` top-level interrupt trap covers SIGINT during the cooldown `Sys.sleep`.

## Store schema

Key `(git_head, cell_id, regime)`. Columns: `run_id, ts_utc, git_head, cell_id, exe, spec_source, measurable, verified, args_str, regime, clock_target_mhz, band_lo, band_hi, attempt, ms, throughput, unit, clock_sm_mhz, clock_mem_mhz, power_w, temp_c, throttle, gpu_mode, valid, reject_reason, rc`. Format: append-only JSONL (source of truth, `append_jsonl_row`) + regenerable RDS rollup (`bench_all_collect.R`, `rbindlist(fill=TRUE)`).

## Invariant preservation

| Invariant | Source | How preserved |
|---|---|---|
| never-abort | bench-all | regime loop outside per-config `tryCatch` (`:603-608`); pwsh continues on non-130 `cellExit` (`:462-466`); always exit 0 |
| retry-to-min-valid | bench-all | `collect_valid_samples` reused verbatim; only `validate_fn` carries `clock_band`; bounded by `max_attempts` |
| two-sided clock-band | grid | `validate_sample(clock_band=c(lo,hi))` (`bench_measure.R:52-61`) starts being *passed*; native passes NULL |
| resume | grid + enhanced | `SUM(valid)>=n_valid` over `(git_head,regime,cell_id)` for measurable; presence-done for non-measurable/skipped |
| Ctrl+C-safe -rgc | grid | lock lifecycle host-side, outside R; both SIGINT routes hit idempotent `Run()` with `LockApplied` true |
| lock sentinel | grid | `.LOCK_HELD` carried verbatim; hard-blocks next start unless `-ForceClearSentinel` |
| parse-miss != failure | shared | once-native non-measurable; three segregated `summary.md` buckets; taxonomy store columns |

## Phased plan

1. **Phase 1 — prove the locked foundation:** P2-5 (single-Ctrl+C mid-sweep) + P2-6 (full 7-kernel grid) on the *unmodified* grid code. The locked path is currently UNPROVEN; convergence cannot rest on it. *(depends on: none — must come first)*
2. **Phase 2 — GPU-free convergence logic:** unified spec, taxonomy×regime gating (incl. the explicit `[native]` default that `grid_measure.R:100` does not provide), store columns, `bench_all_collect.R`. Fixture-testable via `--list`/`--dry-run` + rbindlist diff. *(parallel with Phase 1)*
3. **Phase 3 — orchestrator + R planner/resume:** generalize `run_grid_sweep.ps1` to drive `--regime` per group; add `bench_all.R --regime/--clock/--band/--plan-regimes/--resume` and the corrected resume predicate. New scope: bench_all has only `--list` today. *(depends on: 1 and 2)*
4. **Phase 4 — full-corpus elevated run + RE-TEST the crux** under the new long-child-under-lock unit of work (a long full-corpus child held under one continuous lock is genuinely new vs grid's short per-cell child). *(depends on: 3)*
5. **Phase 5 — retire** `grid_measure.R`/`grid_sweep.yml`/`grid_collect.R`; redirect references to the converged tool. *(depends on: 4)*

## Grafted ideas (from non-winning spines)

- **Taxonomy as store columns** (from store-first): `spec_source`/`measurable`/`verified` on every row; orchestration-independent, makes the rollup self-describing (bench_all already carries `spec_source` in `attempt_row`, `bench_all.R:415`).
- **`SUM(valid)>=n_valid` resume predicate** (from store-first, corrected): fixes grid's presence-based under-sampling (a 3/5-valid cell wrongly skipped → the median lies), with the taxonomy carve-out store-first omitted (presence-done for non-measurable/skipped, which emit 1 or 0 rows and would otherwise loop forever against a sample target).
- **Assert-Elevated gated on `n_locked_groups>0`** (optional hardening): only helps a native-only plan routed through the orchestrator; the core native-elevation-free guarantee already rests on the direct Rscript path.

## Open risks

- **validate_when divergence (numbers, not lifecycle):** A keeps `bench_all`'s `DEFAULT_VALID_WHEN` + real pre/post on the locked path vs grid's minimal `valid_when` + post/post (`grid_measure.R:281-284`). Safe today (`data/baselines.json $default_valid_when` has no `min_clock_sm` — verified) but locked per-(kernel,regime) medians will NOT be bit-identical to grid's prior numbers. The lock *lifecycle* is bit-identical; the fairness *gate* is not. Phase 4 must compare converged numbers against grid's Phase-2 artifact and document any delta.
- **per-kernel `min_clock_sm` hazard:** if any spec sets a boost-calibrated `min_clock_sm` in its `valid_when`, then at a low locked regime (1200/1410) a HEALTHY in-band sample fails the floor → `collect_valid_samples` retries to `max_attempts` → spurious `failed`. No shipped spec does this today; the converged spec MUST carry a spec-authoring guard. A config-authoring hazard, not a structural defect.
- **Apply-Lock setup micro-window (R1, pre-existing, inherited):** `run_grid_sweep.ps1:326` runs `-lgc` BEFORE `LockApplied=$true` at `:327-328` + the sentinel write. A Ctrl+C in that ~3-statement window leaves the GPU physically locked while `GridSweepCleanup::LockApplied` is still false, so `Run()` SKIPS `-rgc` and no sentinel warns the next run. Narrow, lives in lock SETUP (not the retry-inside-lock crux), pre-existing in grid. Fix: set `LockApplied`/sentinel BEFORE issuing `-lgc` (then `-rgc` on a never-actually-locked GPU is a harmless no-op).
- **new planner/resume contract (scope):** `bench_all` today has only `--list` — no `--resume`, no plan handshake. Phase 3 adds a whole grid-style planner subcommand PLUS switches pwsh's plan source. Real new scope, not "two edit sites".
- **results.json regime key:** downstream consumers are tolerant on the RDS side via `fill=TRUE`, but `results.json` is a single pretty-printed object, not JSONL — `fill=TRUE` does NOT cover it. `results.json` readers need the new `regime` key handled explicitly.
- **long locked child vs short cell (unproven):** grid's per-cell child is short; A's per-regime child runs the full measurable corpus under ONE continuous lock; adaptive cooldown `Sys.sleep` now executes under the lock. Bounded and never-stranding, but genuinely new behavior that Phase 4 must validate empirically.
- **Ctrl+C-safe `-rgc` rested on a flawed mechanism (Phase-1 finding, 2026-06-05):** the invariant table above credits grid's "lock lifecycle host-side, both SIGINT routes hit `Run()`". Running #135 P2-5 exposed that this did NOT hold: (a) a Ctrl+C during the child's `renv` autoloader exited 1 (not 130) → sweep continued; (b) a console Ctrl+C is a **process-group** signal that also hits R's `system2` bench wait, which returns **rc=0** — so the child exit code can't drive the abort at all. **Fix landed in `run_grid_sweep.ps1`:** child runs with the renv autoloader bypassed (`--no-init-file` + `R_LIBS_USER`), and `wsl.exe` launches in a **new process group** so only pwsh gets the Ctrl+C and its `CancelKeyPress` handler owns the abort (kill child tree incl. bench, `-rgc`, sentinel) — independent of the child exit code. **The converged Spine A runs this same orchestrator, so its Ctrl+C-safety depends entirely on this Route-B mechanism.** Verified GPU-free (DryRun); the real console-Ctrl+C abort is pending the #135 elevated P2-5 re-run (Phase 1 of the plan). Treat the Q4 "Ctrl+C-safe `-rgc`" row as *contingent on Route-B passing elevated validation*.

## Mapping to #152 acceptance criteria

1. *One schema, regimes default `[native]`, taxonomy kept* → Q1: `bench_all.yml` + merged `clocks`/`defaults`, explicit `[native]` default, taxonomy untouched.
2. *Native lock-free in plain WSL, no elevation tax* → Q2: `make bench-all` direct path + `needsLock=FALSE` native group.
3. *One store keyed (git_head, kernel, regime)* → Q3: JSONL+RDS keyed `(git_head, cell_id, regime)`.
4. *All invariants carried* → Q4 invariant table (never-abort, retry, two-sided band, resume, Ctrl+C `-rgc`, sentinel, parse-miss≠failure).
5. *Reuse cuasmR, no third path* → Q5: same five-stage pipeline, `clock_band` is the single pre-existing seam, `grid_measure.R` retired.

# Grid sweep methodology

Reference for the multi-kernel × clock grid sweep tool delivered by
issue #135. The tool runs each baselined kernel under each meaningful
SM-clock regime, collects raw per-sample throughput data, and persists
results for downstream analysis.

This document covers:

- the architecture and division of labour between the three pieces,
- the canonical resume key,
- guidance on which kernels benefit from clock-locking and which do
  not,
- the failure modes the tool surfaces and how to read them,
- the durable invariants future revisions must respect.

The tool is not a replacement for `bench_regress.R` (which performs
pass/fail gating against `data/baselines.json`) or for
`run_locked_eval.ps1` (the single-cell operator driver from #131).
Each tool has a distinct job:

| Tool                       | Job                                         |
|----------------------------|---------------------------------------------|
| `bench_regress.R`          | Pass/fail vs baseline (pre-push gate, CI).  |
| `run_locked_eval.ps1`      | Single (kernel, clock) cell + log + JSONL row.|
| `run_grid_sweep.ps1` (+ R) | Many (kernel × clock) cells; raw data set.  |

## Architecture

Three pieces, three responsibilities. None reach into another's
scope.

```
  grid_sweep.yml       <-- declarative spec (data, not code)
       |
       v
  grid_measure.R       <-- R brain
       |--- mode=plan       reads YAML + resume JSONL, emits cell list
       |--- mode=measure    measures one cell, appends JSONL rows
       v
  run_grid_sweep.ps1   <-- PowerShell orchestrator
       |--- elevation + sentinel + Ctrl+C handler
       |--- calls planner once
       |--- iterates cells grouped by clock_target
       |--- lock / measure-many / restore per group
       v
  eval_logs/grid_sweep_samples.jsonl   <-- primary store
       |
       v
  grid_collect.R       <-- materialises RDS view from JSONL
       |
       v
  eval_logs/grid_sweep_results.rds     <-- derived analytical view
```

### Why JSONL as the primary store

`saveRDS` / `readRDS` is not atomic. A Ctrl+C between unlink and
rename, or mid-serialise, can corrupt the file. A sweep that runs for
an hour and crashes 50 minutes in must not lose 50 minutes of data.

JSONL appends, one line per sample, are atomic at line boundaries on
both POSIX and Windows for writes below `PIPE_BUF` / page size. A
hard kill leaves at most one truncated final line, which
`grid_collect.R` skips with a one-line warning.

RDS is regenerable from JSONL at any moment. Treat it as a cache.

### Why R is the planner and PowerShell is just the lock driver

The spec is YAML. R has a `yaml` package; PowerShell does not (the
`PowerShell-Yaml` module is an external install we choose not to
depend on). Putting the spec-handling in R keeps the orchestrator
small and stateless: it just iterates a cell list R hands it.

The orchestrator must run on Windows because only the Windows-side
`nvidia-smi.exe -lgc` actually applies a clock lock that propagates
to WSL CUDA. R cannot do that. So we get:

- Spec parsing, validation, planning, measurement, persistence: R.
- Elevation, lock orchestration, child-process lifecycle, Ctrl+C
  safety: PowerShell.

### Why the C# cancel handler

The PowerShell-script-block form of `[Console]::CancelKeyPress`
fails at runtime with `PSInvalidOperationException: no Runspace
available`. The .NET cancel event fires on a worker thread that has
no PowerShell Runspace; script blocks cannot execute there.

Issue #135 Phase 1 (commit `7bd55b4`) hit this in T2 testing — pwsh
crashed mid-sleep and the GPU stayed clock-locked. Fix: implement
the handler in C# via `Add-Type`; pure .NET code runs on the worker
thread without needing a Runspace. The `finally` block in PowerShell
remains the normal-completion path; the C# handler is the cancel
path. Both invoke the same idempotent cleanup.

### Why both kill paths are needed

`Process.Kill(entireProcessTree = true)` walks the Windows process
tree. WSL2 Linux processes are children of `init` inside the WSL VM,
not Windows children of `wsl.exe`, so they are not in that tree and
survive the kill. The C# cleanup therefore also issues
`wsl.exe -e pkill -9 -f grid_measure.R` to clear the Linux side.

## Resume key (canonical)

The resume key is the three-tuple

```
(git_head, clock_target_mhz, cell_id)
```

A row in `grid_sweep_samples.jsonl` is considered "already measured"
for resume purposes if its `(git_head, clock_target_mhz, cell_id)`
matches a planned cell. `--resume-jsonl` plus `-Resume` filters those
cells out of the planner output.

Notes:

- `clock_target_mhz` is `null` in JSON for the native (no-lock)
  regime; the string `"native"` is used in keys.
- `run_id` is **not** part of the key. A resumed sweep gets a new
  `run_id`; the previous run's rows still count.
- A new commit invalidates all prior rows for resume purposes (the
  kernel code may have changed). This is intentional — measuring a
  new commit's kernels against a stale JSONL would mix datasets.
  If you want to share data across commits, query the JSONL directly
  rather than relying on the resume filter.

## Choosing meaningful regimes

Not every kernel benefits from every clock. The `regimes` list in
`grid_sweep.yml` per kernel exists to avoid wasted cells. Use the
table below as a starting point and refine based on
`gpu_reflections.md` observations and the kernel's measured
roofline.

| Kernel type           | Typical bottleneck   | Useful regimes                            |
|-----------------------|----------------------|-------------------------------------------|
| Dense HGEMM 2048-4096 | Tensor-core throughput, bandwidth-balanced | `native, 1605, 1710` — modest sensitivity to clock; locking only useful as a sanity check |
| Sparse INT8 GEMM 4096 | Power-bound at native boost | `1200, 1410, 1500, 1605, 1710` — full plateau map; **no `native`** because every native sample throttles |
| Flash attention seq=1024 | Tensor-core compute-bound, clock-sensitive | `native, 1605, 1710` — wide sensitivity to clock |
| 2D conv (cuDNN-like)  | Mixed                | `native, 1605` — measure once locked + native baseline |

Two guard rails:

1. **If a kernel's bench at native boost ever shows `SwPowerCap`
   throttle**, the kernel is power-bound. Native-regime samples
   under-report throughput in a way that depends on the order of
   throttle events across the bench's averaged launches. The
   sweep records every sample with its throttle state so the bias
   stays auditable, but for headline numbers prefer the highest
   locked clock at which all samples are clean.

2. **If a kernel's throughput is invariant to clock**, it is
   bandwidth-bound or stalled. Lock-sweeping it produces noise.
   Drop low clocks from its `regimes` list once measured.

## Reading the data

Run `grid_collect.R --print` for a quick post-sweep summary:

```
Rscript scripts/probe/grid_collect.R --print
```

The per-`(cell, clock)` table reports `n` (valid samples), median,
min, max, ms median, and the median observed SM clock. The reject
reasons table shows why samples were dropped. Patterns:

- `clock_out_of_band:X not in [lo,hi]` — clock lock not applied or
  drifted. Under `-NoLock` this is expected for any cell with an
  integer `clock_target_mhz`. Under a real lock, an isolated reject
  means the GPU briefly left the band (rare, normal). All-rejected
  means the lock never applied — check elevation, check that the
  Windows-host `nvidia-smi.exe -lgc` was actually run, and inspect
  the sentinel file content if one was left behind.
- `throttle:SwPowerCap` — kernel exceeded the 150 W laptop power
  ceiling during the run. Lock at a lower clock to escape the cap.
- `throttle:HwSlowdown` or `HwThermalSlowdown` — hardware-side
  thermal trip. Cool the laptop, do not raise the lock.
- `parse_failed` — bench output did not match the `match` /
  `value_label` / `section` fields in the YAML. Inspect the cell's
  raw bench output by re-running it under `run_locked_eval.ps1`.
- `rc=N` — bench exited non-zero. Build issue; re-run `make`.

## Failure modes the tool surfaces

| Symptom                            | Diagnostic                          |
|------------------------------------|-------------------------------------|
| All locked-cell samples band-reject | Lock not applied. Elevation? `nvidia-smi.exe -lgc` rejected? Confirm with `nvidia-smi.exe --query-gpu=clocks.current.sm`. |
| One locked-cell sample band-rejects | Normal occasional drift. Increase `band_mhz` in `defaults` if it persists. |
| All samples for one cell `parse_failed` | YAML `match`/`value_label`/`section` don't match this kernel's output. Run the bench by hand and inspect. |
| All samples `throttle:SwPowerCap` for one cell | Power-bound kernel. Lock lower. |
| `Stale lock sentinel present` at startup | Previous run crashed or was hard-killed mid-lock. Manually run `nvidia-smi.exe -rgc` from elevated pwsh, then `-ForceClearSentinel` (or delete the sentinel file). |
| Sweep complete but `grid_sweep_samples.jsonl` empty | Planner found no pending cells (all `already_done`), or every cell errored before its first sample. Check the orchestrator's per-cell `[measure]` log lines. |

## Invariants future revisions must respect

1. **JSONL is append-only.** Never rewrite. Never delete rows.
   `grid_collect.R` and any other reader must tolerate the JSONL
   growing under their feet.
2. **The resume key is `(git_head, clock_target_mhz, cell_id)`.**
   Adding fields is fine. Changing the key requires a migration
   plan.
3. **The C# cancel handler must always restore the clock.** Any
   new code that holds a lock must mirror its state into
   `[GridSweepCleanup]::LockApplied = $true` the instant `-lgc`
   succeeds, and clear it when `-rgc` succeeds. The two paths
   (`finally` and `OnCancel`) must call the same cleanup function.
4. **Sentinel before lock, not after.** A sentinel without a lock
   is recoverable (the next run sees it, refuses to relock, the
   operator clears it). A lock without a sentinel is dangerous
   (the next run sees a clean state, locks again, leaves the GPU
   doubly-clamped).
5. **The orchestrator never writes to the JSONL.** Only
   `grid_measure.R measure` does. This keeps the writer single,
   the atomic-append guarantee single-source-of-truth, and the
   PowerShell side trivially correct.
6. **The single-cell `run_locked_eval.ps1` driver is not
   extended into grid sweep.** Two scripts, two scopes. Conflating
   them re-introduces the "single cell vs many cells" decision into
   every operator-shell run.

## Future work

- **#134 cuasmR migration.** `grid_measure.R` is the canonical
  measurement function. Lands in `scripts/probe/` for now; the
  cuasmR migration absorbs it as a package function with `R CMD
  check` coverage.
- **#128 OC showcase.** Grid sweep above native clock is the OC
  data source. Add `regimes: [1850, 1900, 1950]` for the
  appropriate kernels once an over-volt is dialled in.
- **Cross-host data.** When a second host enters the dataset, the
  resume key may want a `gpu_uuid` component. Currently the JSONL
  records `gpu_uuid` per row but the planner does not key on it.
  Add when needed; do not pre-build.

## Test plan

See the in-script `TEST_PLAN` block in `run_grid_sweep.ps1`. The
canonical matrix:

| Test  | Scope                                      | Status      |
|-------|--------------------------------------------|-------------|
| T1    | -NoLock smoke, no GPU touch                | Phase 1 PASS |
| T2    | Ctrl+C mid-sleep with real lock            | Phase 1 PASS |
| T2c   | Full sleep, finally-path cleanup           | Phase 1 PASS |
| T4    | Stale sentinel refuses to relock           | Phase 1 PASS |
| T4b   | -ForceClearSentinel overrides T4           | Phase 1 PASS |
| P2-1  | -DryRun: plan + groups, no measurement     | Phase 2 PASS |
| P2-2  | -NoLock + -OnlyCellId: end-to-end measure  | Phase 2 PASS |
| P2-3  | Resume: cells in JSONL marked already_done | Phase 2 PASS |
| P2-5  | Ctrl+C mid-measure: child killed, lock restored | TBD (elevated) |
| P2-6  | Full elevated sweep (the real artifact)    | TBD (elevated, ~1 h) |

The TBDs are runtime-only validations and require an elevated
Windows pwsh. They are not blocking for the tool's introduction.

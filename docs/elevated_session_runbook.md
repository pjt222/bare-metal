# Elevated GPU session — runbook

> Prepared 2026-06-04. Turnkey copy-paste for the remaining GPU work that
> needs the **host-side `nvidia-smi.exe -lgc` clock lock** = an **elevated
> (Administrator) Windows PowerShell**. Plain WSL `nvidia-smi -lgc` is denied;
> the host-side lock applies to the whole GPU incl. WSL CUDA.
>
> Repo: `D:\dev\p\bare-metal`. WSL Linux R: `/usr/local/bin/Rscript` 4.6.0.
> Pre-flight already verified (planner parses 28 cells, no stale lock sentinel,
> GPU unlocked P8). This file is session scratch — discard or fold into
> CONTINUE_HERE when done.

## North star (#152)

The end goal is **one** tool that runs **all** kernels against **native and/or a
defined grid of locked clocks** — converging today's two halves: bench-all
(#124, full corpus, native) and the grid sweep (#135, subset, locked grid). See
[#152](https://github.com/pjt222/bare-metal/issues/152). Until that lands, this
runbook drives the two tools side by side: bench-all native for the corpus,
`run_grid_sweep.ps1` / `run_locked_eval.ps1` for the locked clocks.

## What needs elevation (and what does NOT)

| Work | Needs elevated pwsh? | Driver |
|------|----------------------|--------|
| #135 P2-5 single-Ctrl+C re-test | **Yes** (locks clocks) | `run_grid_sweep.ps1` |
| #135 P2-6 full grid sweep (~1h) | **Yes** | `run_grid_sweep.ps1` → `grid_collect.R` |
| #124 native full bench-all (~45 min) | **No** — runs in plain WSL | `make bench-all` |
| #124 fair 4096³ medians (the throttlers) | **Yes** (lock 1605) | `run_locked_eval.ps1` |
| #128 OC showcase | **Yes** (above-native clock data) | grid sweep @1710 / `run_locked_eval.ps1` |

`make bench-all` itself has **no clock lock** — it runs native and just reports
which configs need a locked re-measure. So the #124 bulk can run in WSL anytime;
only the heavy-config fair numbers come from the elevated `run_locked_eval.ps1`.

## Pre-step (run once, in WSL — keeps the full-sweep collection clean)

Archive the stale May-27 partial grid JSONL so `grid_collect.R` reads only the
new run:

```bash
cd /mnt/d/dev/p/bare-metal
mv scripts/probe/eval_logs/grid_sweep_samples.jsonl \
   scripts/probe/eval_logs/grid_sweep_samples.20260527_partial.jsonl
```

## Open the elevated shell

Start menu → type "PowerShell" → right-click → **Run as administrator**.
(`run_grid_sweep.ps1` and `run_locked_eval.ps1` assert Administrator and abort
otherwise — except under `-NoLock`.)

---

## Phase A — #135 P2-5: single-Ctrl+C abort re-test (~3 min)

Verifies the `246c961` fix: one Ctrl+C press aborts cleanly (was needing three).

```powershell
pwsh -File D:\dev\p\bare-metal\scripts\probe\run_grid_sweep.ps1 -OnlyCellId igemm_sparse_4096
```

- **Wait** for the first per-sample line of any clock group, then press **Ctrl+C ONCE**.
- **Expect:** `Bench exited 130 (SIGINT)` → `Cell cancelled by user` → cleanup → exit.
- **Verify no orphans + lock released:**

```powershell
Get-Process Rscript -ErrorAction SilentlyContinue   # expect: nothing
wsl -- pgrep -f grid_measure.R                       # expect: nothing
nvidia-smi.exe --query-gpu=clocks.sm,pstate --format=csv,noheader  # expect P0/P8 boost, not pinned
```

- **If single press still needs multiple** → the fix is incomplete; stop and report (do not proceed to the full sweep). Note exactly how many presses + the last lines printed.
- **Optional dry-run first** (no GPU, validates harness): add `-DryRun`.

---

## Phase B — #135 P2-6: full grid sweep (~1h) + close #135

28 cells × {1200,1410,1500,1605,1710,native}, grouped by clock (one lock per
group). Resumable.

```powershell
pwsh -File D:\dev\p\bare-metal\scripts\probe\run_grid_sweep.ps1
```

- Locks each clock group via `nvidia-smi.exe -lgc`, restores between groups; `finally` + Ctrl+C handler both restore via `-rgc`.
- **If it crashes / you interrupt:** re-run with `-Resume` (skips cells already in the JSONL for the current `git_head`). If it left a stale lock, first run `nvidia-smi.exe -rgc` then re-run with `-ForceClearSentinel`.
- **Materialize + inspect** (WSL):

```bash
wsl -- bash -lc "cd /mnt/d/dev/p/bare-metal && /usr/local/bin/Rscript scripts/probe/grid_collect.R --print"
```

  Reads `eval_logs/grid_sweep_samples.jsonl` → `grid_sweep_results.rds`, prints
  per-(cell,clock) median + reject-reason histogram. Sanity-check against the
  known points (igemm_sparse_4096: 1410→~44k, 1500→~47k, 1605→~50.5k dq-GFLOPS).
- **Close #135** once the plateau map looks complete and Ctrl+C (Phase A) passed.

---

## Phase C — #124 publication-grade numbers

**C1. Native full corpus (no elevation needed — can run in plain WSL):**

```bash
cd /mnt/d/dev/p/bare-metal
make bench-all                       # defaults: --min-valid 5 --max-attempts 15
```

- ~45 min. Writes `results/bench_all/<ts>/{results.json,summary.md,samples.jsonl}`.
- Confirms the #148 fix: flash/resblock/attention_layer `infer` specs should now
  read `ok` (were `failed`/`non-measurable` in the pre-fix run `20260604T103311`).
- Heavy 4096³ configs will land `degraded`/`failed` on `SwPowerCap` — that's
  expected at native; their fair numbers come from C2.
- **Faster confirmation pass** (heavy configs degraded, rest clean, ~20 min):
  `make bench-all ARGS="--max-attempts 6"`.

**C2. Fair 4096³ medians (elevated, lock 1605):** for each throttling config,
e.g.:

```powershell
.\scripts\probe\run_locked_eval.ps1 -ClockMHz 1605 -Kernel kernels/gemm/igemm/igemm_sparse_tiled.cu
```

- Default kernel is `igemm_sparse_tiled.cu` @ 1605 (the `data/baselines.json`
  `clock_lock` entry). Logs to `eval_logs/<stamp>_<kernel>_<mhz>.log` + a row in
  `eval_logs/results.jsonl`. Restores boost via `-rgc` in `finally`.
- Repeat with `-Kernel` for any other heavy config you want a fair median on.

Then consider closing epic **#124** (runner shipped + publication numbers captured).

---

## Phase D — #128 OC showcase (deferred)

Data source = the Phase B grid sweep's **above-native** clock cells (1710), plus
any `run_locked_eval.ps1 -ClockMHz 1710` single-kernel runs. No new code — it's a
presentation/showcase of the highest-clock numbers the lock holds cleanly.
Decide scope after the grid sweep data is in (`grid_sweep_results.rds`).

---

## Safety / recovery (any phase)

- **Always restore boost** if anything aborts mid-lock:
  ```powershell
  nvidia-smi.exe -rgc
  ```
- **Stale lock sentinel** (`eval_logs/.LOCK_HELD`) blocks `run_grid_sweep.ps1`:
  run `-rgc` first, then re-run with `-ForceClearSentinel`.
- **Orphan check** after any interrupt:
  ```powershell
  Get-Process Rscript -ErrorAction SilentlyContinue
  wsl -- pgrep -f 'grid_measure.R|bench_regress.R'
  ```
- **WSL CUDA wedge** (interrupted GPU run spins a bench at ~99% CPU, wedges NVML):
  `kill -9` the spinner tree in WSL; do **not** `wsl --shutdown` from inside WSL.
  Recovery if NVML stays dead = restart the WSL session (Windows-side).

## Suggested order

A (3 min, gate for B) → B (~1h, close #135) → C1 (~45 min, WSL, can overlap) →
C2 (per heavy config) → D (after B data lands).

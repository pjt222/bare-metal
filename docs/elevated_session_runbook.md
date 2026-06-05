# Elevated GPU session — runbook

> Prepared 2026-06-04. Turnkey copy-paste for the remaining GPU work that
> needs the **host-side `nvidia-smi.exe -lgc` clock lock** = an **elevated
> (Administrator) Windows PowerShell**. Plain WSL `nvidia-smi -lgc` is denied;
> the host-side lock applies to the whole GPU incl. WSL CUDA.
>
> Repo: `D:\dev\p\bare-metal`. WSL Linux R: `/usr/local/bin/Rscript` 4.6.0.
> **Pre-flight RE-VERIFIED 2026-06-05 @ HEAD `cc303a5`** (GPU-free, from WSL):
> planner parses **28 cells / 7 kernels** (1605×7, 1710×6, native×6, 1500/1410/1200×3);
> **0 already-done** at this HEAD (the May-27 partial was a different git_head, so the
> sweep runs fresh); all 5 grid bench exes built; GPU unlocked (P8, 210 MHz, idle);
> no stale `.LOCK_HELD` sentinel; the stale partial JSONL has been **archived** to
> `grid_sweep_samples.20260527_partial.jsonl` (Pre-step below already done — skip it).
> This file is session scratch — discard or fold into CONTINUE_HERE when done.
>
> **Design context:** the #152 convergence design is now resolved
> ([`convergence_152_design.md`](convergence_152_design.md), [#152 comment](https://github.com/pjt222/bare-metal/issues/152)).
> Phase 1 of that plan = **prove this locked foundation on the UNMODIFIED grid code**
> = exactly Phases A + B below. Run these first; convergence implementation waits on them.

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

## Phase A0 — non-elevated cancel self-test (run FIRST, in a NORMAL pwsh) (~2 min)

> **Do this before opening the admin shell.** `-NoLock` runs the **real**
> measurement (a real child to interrupt) but applies **no `-lgc`** and **skips
> the elevation assert** — so it exercises the entire new Route-B cancel path
> (`CreateProcess` new-group → `CancelKeyPress` → `KillChild` + bench `pkill` →
> exit 130) with **zero lock-stranding risk**, no admin. This is the actual
> arbiter of "does single-Ctrl+C abort now". If it passes, the locked Phase A
> differs only by Apply-Lock/`-rgc` (trivial). Use a **throwaway `-Jsonl`** so
> test rows don't land in the real store at this `git_head` (the P2-6 resume
> would treat them as already-done). Pick the **cheap** cell (hgemm 2048, no
> throttle/heat).

```powershell
# NORMAL (non-admin) PowerShell:
pwsh -File D:\dev\p\bare-metal\scripts\probe\run_grid_sweep.ps1 `
  -NoLock -OnlyCellId hgemm_16warp_2048 `
  -Jsonl $env:TEMP\p25_selftest.jsonl
```

- Wait for a **`sample N/7`** line, press **Ctrl+C ONCE**.
- **PASS (verified 2026-06-05 — Route A, renv-bypass enabled):**
  ```
  ^CInterrupted by user (SIGINT)
  WARNING: Cell cancelled by user (R exit 130). Aborting sweep.
  Sweep aborted on user cancel.
  ```
  The sweep **stops** on the first press (no `[2/28]`). Mechanism: the child
  reaches the interruptible measurer in ~3s (no renv window), R's top-level
  trap catches the SIGINT → `quit(130)` → pwsh breaks the sweep. (The Route-B
  `[CancelKeyPress]` path is a benign backstop and normally does NOT fire —
  WSL delivers the Ctrl+C to R regardless of the Windows process group.)
- **Verify clean:** `wsl -- pgrep -f /kernels/` → nothing; `wsl -- pgrep -f grid_measure.R` → nothing.
- **FAIL (silent no-op):** if Ctrl+C does **nothing** (no `[CancelKeyPress]`, sample
  lines keep printing), the new process group is shielding the child and the
  cancel mechanism is still broken. **Recovery:** the child runs in its own group,
  so **Ctrl+Break** (not Ctrl+C) reaches it — press that, or close the window;
  then `wsl -- pkill -9 -f /kernels/` and `wsl -- pkill -9 -f grid_measure.R`.
  No GPU lock was applied (`-NoLock`), so nothing to `-rgc`. Report the lines.
- Throwaway store: `Remove-Item $env:TEMP\p25_selftest.jsonl` after.

Only proceed to the elevated phases once Phase A0 aborts cleanly on one press.

> **renv-bypass + measurement are already verified** (2026-06-05, GPU-free + a
> `-NoLock` run to completion: 7 valid native samples, band-reject correct). A0
> isolates the one remaining unknown: the console-Ctrl+C → CancelKeyPress abort.

---

## Open the elevated shell

Start menu → type "PowerShell" → right-click → **Run as administrator**.
(`run_grid_sweep.ps1` and `run_locked_eval.ps1` assert Administrator and abort
otherwise — except under `-NoLock`.)

---

## Phase A — #135 P2-5: single-Ctrl+C abort re-test (~3 min) — **validates a new fix**

> **2026-06-05 — the abort itself is already VERIFIED** by Phase A0 (`-NoLock`,
> single Ctrl+C → clean abort via Route A; the renv bypass was the load-bearing
> fix). This elevated Phase only confirms the **locked** variant: that the same
> abort also runs `-rgc` and clears the sentinel so the GPU is left unlocked.
> Background on the two bugs P2-5 exposed + their fixes is in
> `docs/convergence_152_design.md` (open-risks) and the commit
> `grid-sweep: robust single-Ctrl+C abort`.

```powershell
pwsh -File D:\dev\p\bare-metal\scripts\probe\run_grid_sweep.ps1 -OnlyCellId igemm_sparse_4096
```

- **Wait** for a **per-sample line** of any locked group (now ~3s in, not ~26s),
  then press **Ctrl+C ONCE**.
- **Expect (Route A — as seen in Phase A0, plus the lock release):**
  ```
  ^CInterrupted by user (SIGINT)
  WARNING: Cell cancelled by user (R exit 130). Aborting sweep.
  Sweep aborted on user cancel.
  [cleanup] restoring default clock policy (-rgc)...
  [cleanup] -rgc OK
  [cleanup] sentinel cleared
  ```
  The sweep stops on the **first** press (no advance to the next clock group),
  and the `finally` cleanup releases the lock. (`[CancelKeyPress]` lines are a
  benign backstop and normally do NOT appear.)
- **Verify no orphans + lock released:**

```powershell
wsl -- pgrep -f grid_measure.R                        # expect: nothing
wsl -- pgrep -f /kernels/                             # expect: nothing (bench killed)
nvidia-smi.exe --query-gpu=clocks.sm,pstate --format=csv,noheader  # expect P0/P8 boost, not pinned
Test-Path D:\dev\p\bare-metal\scripts\probe\eval_logs\.LOCK_HELD    # expect: False
```

- **If the sweep continues to the next clock group, or `-rgc OK` / `sentinel
  cleared` do NOT appear, or `.LOCK_HELD` is True / the clock stays pinned** →
  the locked abort is incomplete. **Immediately `nvidia-smi.exe -rgc`**, then
  `wsl -- pkill -9 -f /kernels/`, `wsl -- pkill -9 -f grid_measure.R`, and
  re-check `.LOCK_HELD`. Report the exact lines. (If a press is ever a no-op,
  **Ctrl+Break** reaches the child.)
- **Optional dry-run first** (no GPU, exercises the launch path): add `-DryRun`.

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

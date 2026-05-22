# Benchmark methodology — throttle, power, and reproducibility

How to get comparable benchmark numbers out of a laptop GA104 GPU,
and why the obvious levers do not work. Companion to
[`gpu_reflections.md`](gpu_reflections.md) (kernel-level observations)
and the harness scripts under `scripts/bench/`.

Recorded 2026-05-21 on the RTX 3070 Ti Laptop GPU, sm_86,
driver 595.97, CUDA 13.2, under WSL2.

## The problem

A laptop GPU does not hold a fixed clock. Under sustained load it
boosts, hits a limit, and clocks back down. Two consecutive runs of
the same kernel can therefore land at different clock states and
report different throughput — neither number is wrong, but they are
not comparable to each other or to a baseline recorded earlier.

`scripts/bench/bench_regress.R` already refuses to compare a run
against a baseline when the GPU was throttled during the run (see
`bench_meta.R::classify_meta`). That keeps the regression gate
honest, but it means a throttled run is *skipped* — and a full
"run everything" pass needs a way to not silently drop runs.

## Throttle taxonomy

`nvidia-smi` reports an active-throttle bitmask, decoded in
`bench_meta.R::decode_throttle`. The states that make a measurement
unfair (`.UNFAIR_THROTTLES`):

| Reason | Meaning |
|---|---|
| `SwPowerCap` | Driver clamped clocks — kernel asked for more power than the board limit. The dominant one on this machine. |
| `SwThermalSlowdown` / `HwThermalSlowdown` | Clocks cut to hold temperature. |
| `HwSlowdown` / `HwPowerBrakeSlowdown` | Hardware emergency clamp (power brake, critical temp). |
| `ApplicationsClocksSet` | Clocks pinned by an explicit application-clocks request. |

`GpuIdle` is benign (it just means the GPU was idle at the instant of
the snapshot, e.g. between launches) and is allowed.

## Power: the 150 W wall

Measured with `nvidia-smi -q -d POWER`:

```
Current Power Limit : 150.00 W
Default Power Limit : 115.00 W
Max Power Limit     : 150.00 W
Min Power Limit     :   1.00 W
```

Findings:

- The GPU runs at **Current = Max = 150 W**. It is already pinned to
  the ceiling. Default is 115 W, so an OEM performance mode (Lenovo
  Vantage / Legion) or Dynamic Boost has already raised it 115 → 150.
- **150 W is the VBIOS hard cap.** `nvidia-smi -pl` accepts only
  values in `[Min, Max]` = `[1, 150]`. There is no headroom to raise.
- **A larger PSU does not help.** The wall adapter feeds the whole
  platform (CPU, GPU, display, charging). The GPU *board* power limit
  is set in firmware and is independent of adapter wattage. A 300 W
  adapter does not unlock a 150 W GPU.
- **Lenovo Vantage cannot exceed 150 W either.** Its performance mode
  already did its job (115 → 150). The only thing to keep set is
  Performance mode so the limit does not fall back to the 115 W
  default and so the fans run high (which keeps thermal throttle from
  stacking on top of power throttle).

Conclusion: `SwPowerCap` during heavy kernels is real physics. At
150 W the heaviest kernels still demand more than the board will
deliver, and the driver caps them. Power throttle cannot be bought
or configured away on this machine.

## Reproducibility levers

Since the power cap is fixed, comparable numbers come from removing
*clock variance*, not from adding power.

### Clock locking — NOT available on this machine

`nvidia-smi -lgc <min>,<max>` pins the graphics clock to a fixed
range; `-rgc` resets it. Locking the clock low enough that the
heaviest kernel stays under 150 W would make `SwPowerCap` never fire
and every bench run at an identical clock.

**On this machine it does not work.** `scripts/probe/probe_clock_lock.R`
(verdict 2026-05-21) attempted to lock the SM clock and got:

```
-lgc exit status : 255
  Unable to set GPU locked clocks "(gpuClockMin 1095, gpuClockMax 1095)"
  for GPU 00000000:01:00.0: Unknown Error
```

The WSL2 GPU passthrough rejects clock control regardless of root
privilege. `-lgc` is therefore **not a usable lever here** — cooldown
+ retry is the only one (see below). This closes issue #125.

A bare exit code is not proof a lock took effect: WSL can accept the
command syntactically and silently no-op. Any future re-probe must
*set* a clock and *read it back* (`clocks.current.sm`) — which is
what `probe_clock_lock.R` does.

### Cooldown — the only available lever

Insert an adaptive wait between benches: after a bench, poll
`capture_gpu_state()` until temperature and power fall back toward
idle and no throttle is active, with a max-wait cap.

- No privilege, WSL-safe, portable.
- Only resets state *between* benches. A single long kernel
  (e.g. 4096³ GEMM running 100 iterations) can still throttle
  *mid-run*; cooldown cannot fix that — the per-config retry loop
  handles it instead.

Cooldown does not remove *boost variance* — only clock-locking
could, and that is unavailable here. In practice the laptop GA104
settles to a sustained ~1410 MHz cold-clock after warmup and never
reaches the 1785 MHz boost bin in a short single-process run, so a
warmed-up bench is already at a fairly stable clock. Baselines are
recorded at that sustained cold-clock; see
[`rebaseline_protocol.md`](rebaseline_protocol.md).

## GPU mode: hybrid vs dGPU

The laptop has a MUX. In **hybrid** mode the iGPU drives the display
and the dGPU renders/copies; in **dGPU mode** the panel connects
directly to the dGPU.

- dGPU mode removes the Optimus copy path → no copy-jitter, the dGPU
  stays warm and active, fewer power-state transitions mid-bench.
  Good for measurement consistency.
- dGPU mode does **not** raise the 150 W cap. Some Legion SKUs report
  a higher TGP ceiling in dGPU + Performance mode — re-check
  `nvidia-smi -q -d POWER` after switching, but do not assume it.
- **WSL2 cannot observe the MUX state.** `nvidia-smi` under WSL
  reports `Display Active: Disabled` / `Display Attached: No` in
  *both* modes, because the dGPU never has a guest-side display in
  the WSL VM. `GPU Operation Mode` is empty (a Tesla-only field).
  GPU mode therefore cannot be auto-detected from inside WSL.

### Recording GPU mode (issue #126)

Because the mode cannot be auto-detected, it is supplied explicitly.
Set the `BARE_METAL_GPU_MODE` environment variable before a benchmark
run:

```bash
export BARE_METAL_GPU_MODE=dgpu     # or: hybrid
```

`scripts/bench/bench_meta.R` (`capture_gpu_state()`) records it as
`$host$gpu_mode` in every run's metadata, and the one-line GPU-state
header printed by `bench_regress.R` shows `gpu_mode=...`. Accepted
values are `hybrid` and `dgpu`; anything else, including unset,
records as `unknown`. The value is **never** guessed from
`display_active` — that field is identical in both modes under WSL.

## Overclocking — deferred

OC / GPU overdrive (Lenovo Vantage or a third-party tool) is
**deferred**, not adopted, for the corpus benchmark pipeline:

- OC raises clocks at a given power, so the GPU demands *more* power
  and hits `SwPowerCap` *sooner and harder*. It does not lift the
  150 W cap. More throttle, more variance — the opposite of what a
  comparable dataset needs.
- An unstable overclock produces real kernel crashes, which would be
  indistinguishable from genuine kernel bugs and would poison the
  dataset.

OC is a peak-number tool, not a sweep tool. A future, separate task
may use OC to showcase a *single* kernel's headline number, with its
own clearly-labelled measurement context — never mixed into the
comparable corpus.

## Methodology for a full "run everything" pass

The principle for an on-demand full-corpus run is **skip nothing,
record everything**:

- Build the whole corpus first (`make all`) so there are no
  "executable not found" skips.
- Per `(bench, size)` config, retry up to `MAX_ATTEMPTS` times until
  `MIN_VALID` clean samples are collected; classify each attempt as
  `valid` / `throttled` / `failed` from its pre/post GPU state and
  exit code; cool down between attempts.
- A config that cannot reach `MIN_VALID` is recorded as `degraded`
  (with the reason), never dropped. A config that fails every
  attempt is recorded as `failed`. The runner never aborts the
  corpus on a single bench failure.
- The output keeps *every* attempt plus a per-config summary, so
  "no failures" becomes a property the reader verifies from a
  complete report rather than a guarantee baked into the run.

Run-level metadata to capture: ISO timestamp, git commit + dirty
flag, host, OS / WSL build, `nvcc` and driver versions, GPU name +
SM count, clock-lock state, **GPU mode (hybrid / dGPU)**, and the
per-attempt `capture_gpu_state()` snapshots.

This methodology is the design basis for the planned `make
bench-all` target — see the benchmark-pipeline hardening issues on
GitHub.

For the concrete step-by-step procedure to re-record a suspect
baseline (currently `hgemm` + `igemm_sparse`, recorded under a
power-supply fault), see
[`rebaseline_protocol.md`](rebaseline_protocol.md).

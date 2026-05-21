# Re-baseline protocol — hgemm + igemm_sparse

> Status: draft handoff, 2026-05-21 | Owner: USER (controlled
> measurement session) | Unblocks: push of 5 queued `main` commits,
> issues #129 / #125.

## Why this exists

`data/baselines.json` was recorded 2026-05-10. The recording session
hit a **power-supply fault that was fixed mid-first-kernel**. The
first kernel in the file is `hgemm` — so its baseline was captured
under bad power, and `igemm_sparse_tiled` (recorded right after)
inherits the same suspicion.

Symptoms confirming the baseline is wrong, not the kernels (measured
2026-05-21):

| Kernel              | 2026-05-21 spread        | vs baseline        | Verdict                |
|---------------------|--------------------------|--------------------|------------------------|
| `hgemm` 2048³       | 24674–27158 GFLOPS, 5 runs | uniformly 77–85 % | baseline recorded high |
| `igemm_sparse` 2048³| 36800–39500 GFLOPS, 5 runs | uniformly 116–125 %| baseline recorded low  |
| `igemm_sparse` 4096³| 15778–24412 GFLOPS, 5 runs | bimodal 1.55× span | partly real, partly artifact |

`igemm_sparse` 2048³ sitting *uniformly above* baseline is the tell:
a kernel cannot get 20 % faster by itself. The 2026-05-10 number was
recorded low under degraded power.

**Decision (carried from the 2026-05-21 handoff): do not widen
tolerances to paper over this — re-baseline.** Tolerance band-aids
make the regression check lie; a correct baseline makes it tell the
truth.

## Recording clock — decided (#125 closed)

The #125 clock-lock probe ran 2026-05-21. Verdict: **`-lgc` is
rejected** — WSL2 passthrough returns exit 255 ("Unable to set GPU
locked clocks ... Unknown Error") regardless of root privilege.
Clock-locking is **not a usable lever** on this machine.

Therefore baselines are recorded at the **sustained cold-clock** the
laptop GA104 naturally settles to after warmup — measured ~1410 MHz
(43 W, 50 °C, no throttle). This is a laptop-clock baseline, not a
boost-clock one; the 1785 MHz boost bin is unreachable in a short
single-process `bench_regress` run.

Record the recording clock in the `baselines.json` top-level `note`
so future readers know the regime.

## Recording procedure

For each kernel/config in the table below:

1. **Power.** Confirm AC is connected and the PSU is stable (the
   2026-05-10 fault is the entire reason for this exercise). The
   `capture_gpu_state()` host snapshot records AC state; verify it
   reads `ac` not `battery`.
2. **Build the whole corpus** so no config is skipped for a missing
   executable:
   ```bash
   make clean && make all
   ```
3. **Pre-warm.** Run the target bench ~30 times back-to-back (or
   until the SM clock reading stabilises near the sustained
   ~1410 MHz cold-clock) so the measurement is in steady state, not
   cold. No clock-locking — #125 verdict: `-lgc` rejected.
4. **Sample.** Collect **at least 7 valid samples** per config
   (more than the existing "5 valid samples" convention — the bimodal
   `igemm` 4096³ needs the extra resolution). A sample is *valid* only
   if `classify_meta` does not flag it (no throttle, AC). Discard
   SKIPPED samples; keep collecting until 7 valid ones land. Record
   the SM clock of each sample — they should cluster near 1410 MHz;
   a sample far below that is cold and should be discarded as
   unrepresentative even if `classify_meta` does not yet flag it
   (the `min_clock_sm` gate is added *after* this run — see below).
5. **Record the median**, not the mean — the median is robust to the
   `igemm` 4096³ bimodal tail.

### Configs to re-record

Only the four flagged configs. Leave every other kernel in
`baselines.json` untouched — they are not under suspicion.

| File                                         | Config              | Old (2026-05-10)        | Field   |
|----------------------------------------------|---------------------|-------------------------|---------|
| `kernels/gemm/hgemm/hgemm_16warp.cu`         | `2048_2048_2048`    | 0.539 ms / 31875 GFLOPS | `gflops`|
| `kernels/gemm/hgemm/hgemm_16warp.cu`         | `4096_4096_4096`    | 4.327 ms / 31765 GFLOPS | `gflops`|
| `kernels/gemm/igemm/igemm_sparse_tiled.cu`   | `2048_2048_2048`    | 0.544 ms / 31588 TOPS   | `tops`  |
| `kernels/gemm/igemm/igemm_sparse_tiled.cu`   | `4096_4096_4096`    | 4.449 ms / 30889 TOPS   | `tops`  |

`flash_attn_br16_regpv` `1024_8_8` carries the same laptop-clock
artifact (handoff §"Push blocked"). Re-record it in the same session
at the sustained cold-clock and narrow its `tolerance: 0.30`
band-aid back toward the default 0.10 — a same-clock baseline no
longer needs the wide band.

## Output: the `baselines.json` patch

For each config, update `ms` + the throughput field to the new
median, and rewrite the `note` to state: recording date, sample
count, the observed sustained clock, and that it supersedes the
power-fault 2026-05-10 value. Example shape:

```json
"2048_2048_2048": {"ms": <new>, "gflops": <new>,
  "match": "hgemm_16warp (128x128 2blk/SM)",
  "note": "median of 7 valid samples, 2026-05-2X, sustained cold-clock ~1410 MHz (#125: -lgc rejected by WSL, no clock-lock). Supersedes 2026-05-10 / 31875 — that recording hit a power-supply fault."}
```

Also update the top-level `recorded_date`, `previous_recorded_date`,
and `note` to describe the new recording regime.

## After re-baselining

1. **Set `min_clock_sm` (resolves #129).** With the recording clock
   now known and fixed, add `min_clock_sm` to `default_valid_when`
   (or per clock-sensitive kernel) at a floor just below the
   recording clock — e.g. recording at 1410 MHz → `min_clock_sm:
   1380`. Any future run below that floor then reports `SKIPPED`,
   not a false `REGRESSION`. This value **cannot** be chosen before
   this protocol runs — it is the recording clock minus a small
   margin.
2. **Narrow the tolerance band-aids.** With a correct same-clock
   baseline, the `tolerance: 0.30` overrides on `igemm_sparse_tiled`
   4096³ and `flash_attn_br16_regpv` exist only to absorb the
   bimodal `igemm` tail — review whether 0.30 is still needed or can
   drop toward 0.15.
3. **Push.** The pre-push `bench_regress.R` should pass once the
   baseline matches the hardware. This fires `Closes #127` and
   unblocks the queued `main` commits.

## Related

- `docs/benchmark_methodology.md` — throttle taxonomy, the 150 W
  wall, cooldown as the only reproducibility lever.
- Issue #125 — clock-lock probe. **Closed 2026-05-21**: `-lgc`
  rejected by WSL passthrough; baselines record at the sustained
  cold-clock instead.
- Issue #129 — `valid_when` `min_clock_sm` gate (resolved by step 1
  of "After re-baselining").
- Issue #124 — `bench-all` runner, the eventual home of this
  procedure as an automated mode.

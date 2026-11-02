# #195 conv2d variance campaign: pre-registration

Written and committed **before** the run starts. Nothing here may be
changed after the data exists. Deviations go in a separate, dated section
below the results.

## Command

```bash
Rscript scripts/probe/conv2d_variance.R --n 15 --gap 90 --block 1 --burnin 2 \
  --jsonl results/probe/conv2d_variance.jsonl
```

- Harness: `scripts/probe/conv2d_variance.R` at the commit that adds this
  file. It includes the #212 fixes (degenerate designs refused; `--burnin`)
  and the #213 fix (both power references, loadavg and a `snapshot` flag
  per row, and a stop on a missing schema field).
- Executable: `kernels/convolution/conv2d/bench_implicit_gemm`, built
  2026-05-21; `make -n` reports it up to date with its sources.
- Sequence: `WW` + `ABAB…` (15 A, 15 B), so 32 samples x 6 measurements =
  192 rows. Condition B idles 90 s before each sample; A and W do not.

## Regime and host

- Power envelope `150 / 115 / 150 W`, read immediately before the run.
- Peer sessions on this host (`career`, `krems`, `fract-mono`) confirmed
  they would hold builds, tests, R/Python jobs, subagents and GPU use for
  the window. The bare-metal session itself starts no agents and no
  pushes while the run is going.
- The run is bracketed by a power read, `/proc/loadavg` and the top CPU
  processes, before and after. **No polling during the run**: the gaps
  occur only in condition B, so an external poll would perturb B and not A,
  manufacturing the very contrast under test.

## Void rules (carried over unchanged from the August pilot)

A sample is **VOID** if any of these hold:

- `post_power_limit_w != 150`
- a non-`GpuIdle` throttle reason in either snapshot
- `post_clock_sm < 1600`

If **2 or more** of the 30 A/B samples are void, the whole run is void.
W samples are not counted toward that limit.

Caveat, recorded in advance: the snapshots bracket the process from
outside, so `throttle == GpuIdle` cannot show that the timed window was
clean. It can only fail a sample, never certify one.

## Analysis plan

1. Exclude `burnin == TRUE` rows from every A/B analysis.
2. Per measurement (`sd64/sd32/sd128 x explicit/implicit`), regress
   throughput on `order_in_run` and `pre_pstate`, with `condition` as a
   third term. The question is whether dispersion follows **position in
   the process** (the pilot's reading: per-config cold entry), the
   **config**, or the **A/B condition**.
3. Report every measurement row by row. No pooled single number unless
   all rows share a sign.
4. Use per-sample `pre_load_1m`/`post_load_1m` as a covariate to check for
   host contamination.
5. The W rows against the first A/B samples measure the lead-in transient
   directly.
6. The pilot's SD64-specific hypothesis is dead and is not tested.

## Publication rule

Post to #195 only if the run is not void. A void run is still committed,
with the reason it was void.

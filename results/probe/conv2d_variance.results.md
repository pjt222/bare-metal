# #195 conv2d variance campaign: results

Pre-registration: `conv2d_variance.prereg.md`, committed in e7d90df before
the run. Data: `conv2d_variance.jsonl` (192 rows) and
`gate_conv2d_history.tsv`, a committed extract of the gitignored gate log.
`Rscript scripts/probe/conv2d_variance_analysis.R` regenerates every
computed number, Parts 1–4. Not regenerated: the host bracket values (from
`top` and `/proc/loadavg`, recorded at run time) and the pilot figures
(from ac95a65).

**This file was corrected after review round 1 on #224.** Several first-draft
conclusions said more than the data shows. The log at the end lists each
change, and the version first posted to #195 is c6b9f40.

## Run

| | |
|---|---|
| run_id | `20261102T110558`, harness at e7d90df |
| window | 2026-11-02 11:05:54 to 11:38:51 CET, exit 0 |
| design | `WW` + `ABAB…` (15 A, 15 B), `--gap 90 --block 1 --burnin 2` |
| power, before / after | `150 / 115 / 150 W` / `150 / 115 / 150 W` |
| host, before | loadavg 3.17 (lagging a cuasmR install a minute earlier); `top`: 92% idle, no process above 7.5% CPU |
| host, after | loadavg 1.03; `top`: 90% idle |
| peers | `career`, `krems` and `fract-mono` confirmed they were holding heavy work for the whole window |

**Valid: 0 of the 30 A/B samples are void**, against a limit of 2. Every
sample has `post_power_limit_w = 150`, only `GpuIdle` throttles and
`post_clock_sm ≥ 1620` (the minimum, at sample 21). On every row the
power references, loadavg and both snapshots are present and non-NA. That
is #213's acceptance criterion, shown on a real run.

**What "valid" does not mean.** The snapshots are taken outside the timed
window. `GpuIdle` before and after a sample cannot show that the window
itself was free of throttling. Today's own pre-push gate saw `SwPowerCap`
at 150 W on another kernel, so in-window throttling does happen. The rules
can only reject a sample, never certify one.

## Deviations from the pre-registration

All of these could have been seen before the run. They are recorded here
rather than silently worked around.

1. **`order_in_run` cannot be a regressor within a measurement.** It is
   constant for each measurement (`sd64_explicit` is always 1, and so on).
   Position in the process can only be compared *across* measurements, and
   there it is fully confounded with the config: the bench runs its 4
   configs in a fixed order (8 measurements, 6 parsed).
2. **Condition and `pre_pstate` are the same variable in this run.** All 15
   A samples entered at P0 (1770–1785 MHz) and all 15 B samples at P8
   (210 MHz). A regression with both terms is not estimable.
3. **The pre-registered regression was replaced** by Welch tests on means,
   per measurement. That is the estimable part of item 2 of the plan.
4. **`post_load_1m` was not analysed** in the first draft, although the
   pre-registration names it. It is now (Part 4c).
5. **The pre-registration said the SD64-specific hypothesis "is dead and is
   not tested".** Yet this write-up's headline is the SD64 spread. That
   spread is reported because `sd64_implicit` is the gate's config. It is
   post hoc emphasis, not a pre-registered test.

## Design limitation: the A/B manipulation probably never reached the timed window

Found by review, after the run. The bench runs `cpu_conv2d_nhwc_ref`, which
takes seconds of CPU, **before every config's** timed section
(`bench_implicit_gemm.cu:212-216`; explicit timed at `:275-289`, implicit
at `:291-300`). The reference is skipped only for inputs over 16 MB. All
four configs are under that: SD64 1.3 M floats, SD32 0.66 M, SD128 2.6 M,
Small 0.26 M, against a 4.2 M limit. A process takes about
19 s, and its timed GPU work is well under 1 s. So the GPU idles before
every timed section, whatever state it entered the process in. The 90 s
gap changes the state at process **entry**, and that state is very likely
gone by the time the timing starts. A null A/B result is what this design
should produce whether or not the gap matters.

Also, condition B is not the gate's condition for conv2d. `bench_regress.R`
has no sleep, and it runs conv2d straight after `flash_attn`, about 21 s
apart. Condition A is the closer analogue.

## Results

### Per measurement × condition

A: back-to-back, entered P0. B: 90 s idle, entered P8.

| measurement | ord | cond | mean | sd | rel sd | min | max | max/min |
|---|---|---|---|---|---|---|---|---|
| sd64_explicit | 1 | A | 6229 | 172 | 2.76% | 5927 | 6604 | 1.114 |
| sd64_explicit | 1 | B | 6189 | 427 | 6.90% | 4731 | 6611 | 1.397 |
| **sd64_implicit** | 2 | A | 7549 | 435 | **5.76%** | 6924 | 8043 | 1.162 |
| **sd64_implicit** | 2 | B | 7513 | 434 | **5.78%** | 6788 | 8107 | 1.194 |
| sd32_explicit | 3 | A | 4127 | 523 | 12.67% | 3053 | 4411 | 1.445 |
| sd32_explicit | 3 | B | 4344 | 105 | 2.41% | 3976 | 4405 | 1.108 |
| sd32_implicit | 4 | A | 4770 | 48 | 1.00% | 4630 | 4825 | 1.042 |
| sd32_implicit | 4 | B | 4797 | 28 | 0.59% | 4736 | 4830 | 1.020 |
| sd128_explicit | 5 | A | 3601 | 134 | 3.72% | 3123 | 3674 | 1.176 |
| sd128_explicit | 5 | B | 3629 | 33 | 0.90% | 3559 | 3674 | 1.032 |
| sd128_implicit | 6 | A | 6778 | 65 | 0.95% | 6640 | 6893 | 1.038 |
| sd128_implicit | 6 | B | 6769 | 65 | 0.95% | 6619 | 6869 | 1.038 |

Throughput in GFLOPS. The explicit-kernel spreads are each driven by one
or two low samples: `sd64_explicit` B 4731 (sample 30), `sd32_explicit` A
3053, and `sd128_explicit` A 3123.

### Condition (= entry state) effect, A − B, Welch t-test

| measurement | A − B | % of B | 95% CI | p |
|---|---|---|---|---|
| sd64_explicit | +39 | +0.6% | [−210, +289] | 0.74 |
| sd64_implicit | +35 | +0.5% | [−290, +360] | 0.83 |
| sd32_explicit | −217 | −5.0% | [−510, +77] | 0.14 |
| sd32_implicit | −28 | −0.6% | [−57, +2] | 0.066 |
| sd128_explicit | −29 | −0.8% | [−104, +47] | 0.43 |
| sd128_implicit | +9 | +0.1% | [−39, +57] | 0.70 |

No measurement shows a detectable effect: every interval contains 0. The
signs are mixed (3 positive, 3 negative), so no single summary number
applies.

### Lead-in (W samples)

`W1` entered P8 and `W2` entered P0. Only `sd64_implicit` stands out at W1
(6659, the lowest value of all 32 samples). The other five measurements are
normal at W1 and W2. One sample cannot establish a lead-in transient, and B
samples enter P8 too without being low (B4: 8107).

### Host load and drift, within condition (n = 15 each)

- **Load is not a steady fall.** `pre_load_1m` ran from about 1.0 at
  sample 8, up to 3.36 at sample 16, then down to 0.79 at sample 30.
  cor(load, sample index) = −0.31, so load and drift *can* be separated.
  The first draft said they could not.
- **Throughput vs `pre_load_1m`** ranges from r = −0.53 to +0.53 across
  measurements, with mixed signs. The +0.53 is the headline measurement,
  `sd64_implicit` in condition A (p = 0.042). See Part 4a below.
- **Throughput vs `post_load_1m`** (Part 4c; omitted from the first draft)
  is also mixed. The one strong value, `sd32_implicit` in A at r = −0.86,
  is driven by sample 15 (`post_load_1m` 4.56, which also gave that
  measurement's minimum of 4630).
- **Throughput vs `sample_idx`** ranges from −0.38 to +0.60. The three
  tightest measurements (rel sd ≤ 1%: `sd32_implicit`, `sd128_implicit`,
  `sd128_explicit` in B) drift upward.

### Tail events (Part 4d)

These are single measurements more than 10% below their condition's mean
(the mean excludes the sample itself). This is the shape of the
push-blocking drop #195 is about, and the first draft left it in a
footnote:

| sample | cond | measurement | GFLOPS | vs condition mean |
|---|---|---|---|---|
| 11 | A | sd32_explicit | 3100 | −26.2% |
| 13 | A | sd32_explicit | 3207 | −23.5% |
| 31 | A | sd32_explicit | 3053 | −27.4% |
| 30 | B | sd64_explicit | 4731 | −24.8% |
| 5 | A | sd128_explicit | 3123 | −14.1% |
| 20 | B | sd64_implicit | 6788 | −10.3% |

Five of the six are in the **explicit** kernels. In each config, the
explicit kernel is timed first after the CPU reference has left the GPU
idle, so it runs through the clock ramp. That fits the per-config
cold-entry mechanism recorded with the pilot. The gate's 5935 reading
(−26%) is an *implicit* reading, and this campaign recorded nothing that
deep in `sd64_implicit`: its minimum is 6788.

## What this run shows, and what it cannot

The gate measures `conv2d_implicit_gemm` as `sd64_implicit`: the same
executable, section `SD 64`, match `Implicit (single kern)` (see
`data/baselines.json`).

1. **The spread does not depend on what happened before the process
   started.** `sd64_implicit`'s rel sd is 5.76% after a back-to-back start
   and 5.78% after a 90 s idle, and the means do not differ (+0.5%,
   95% CI −3.9% to +4.8%). Because of the design limitation above, this
   does **not** test the gate's cadence. Within the process, the bench idles
   the GPU before every timed section either way.
2. **`sd64_implicit` spreads at a fixed 150 W regime:** rel sd 5.7%,
   range 6788–8107 (1.19×). Its implicit siblings stay within 0.86%
   (`sd32`) and 0.94% (`sd128`).
3. **The campaign did not reproduce the push-blocking tail for this
   measurement.** 0 of 30 `sd64_implicit` samples fell below 6788, while
   3 of 28 gate readings did (5935, 6577, 6751; Fisher p = 0.11). The tail
   #195 is about was not sampled here. It *was* sampled in the explicit
   kernels (table above).
4. **Whether the spread comes from SD64's shape or SD64's position cannot
   be told from this design.** SD64 is always the first config in the
   process (deviation 1).

All 30 samples clear the gate's 10% floor below the 7150 baseline (6435).
The baseline is a median of 5 from 2026-05-10 and sits about 5% below
today's mean of 7531. For why re-baselining it needs care, see #219.

## Pilot claims, checked against this run (n = 30 vs the pilot's 6)

| pilot claim (ac95a65, 2026-08-13) | this run |
|---|---|
| (b) warm entry, P0, is +8.5% faster for `sd64_implicit` | **not replicated**: +0.5%, CI [−3.9%, +4.8%]. Expected under the design limitation, so this does not show that entry state is irrelevant inside the window |
| (a) `sd64_implicit` is about 5× more dispersed than its siblings | **replicated**: 5.67% vs 0.86% / 0.94% |
| explicit-family dispersion falls monotonically with order (23.8 → 12.6 → 2.2%) | **not replicated**: 5.16 → 9.13 → 2.68% |

## Exploratory (NOT pre-registered)

**Bimodality: consistent with a single mode** (Part 4b). The largest gap in
the sorted `sd64_implicit` values is 166 GFLOPS (7611 to 7777). Under a
single normal distribution with the same n, mean and sd, a gap at least
that large occurs in 92% of draws. The observed max-gap/mean-spacing ratio
is 3.65, against a null median of 4.88. The first draft's "two clusters"
reading is withdrawn.

**Covariates, high vs low split at 7694** (Part 4a; uncorrected p,
about 12 variables examined):

| variable | high − low | 95% CI | p |
|---|---|---|---|
| `pre_load_1m` | +0.68 | [+0.17, +1.18] | 0.011 |
| `post_load_1m` | +0.42 | [−0.16, +1.00] | 0.15 |
| `sd32_explicit` | +256 | [−14, +526] | 0.061 |
| `sd64_explicit` | +157 | [−72, +387] | 0.17 |
| `sd128_explicit` | −40 | [−121, +40] | 0.30 |
| `sd128_implicit` | +25 | [−24, +74] | 0.31 |
| `sd32_implicit` | +8 | [−23, +38] | 0.61 |

Condition (A 8 high / 7 low, B 6 / 9), `post_clock_sm` (1760 vs 1759),
`post_temp_c` (57.4 vs 57.4) and `post_power_w` (66.7 vs 67.5) do not
differ.

**Host load is the one variable nominally associated with the high
samples.** It does not survive a correction for about 12 comparisons. Its
direction, more load with *higher* throughput, is the opposite of
contamination. The first draft's "no recorded variable separates the two
groups" was wrong: it dropped this row.

The cause is not located. Having no outside correlate does **not** put it
inside the timed window. Per-process state, or the timing of the clock
ramp between the CPU reference and the timed section, fit the data
equally well.

## Gate history (exploratory; `gate_conv2d_history.tsv`, Part 4e)

The gate runs conv2d last, straight after `flash_attn`, with no sleep. It
comes after `igemm_pipelined_cpasync`, which hits `SwPowerCap` at about
130 W and 57–74 °C. The extract holds 28 non-throttled readings of this
config (2026-07-31, 2026-08-12 and 2026-11-02, including five from today's
later pushes):

`5935 6577 6751 6908 7043 7052 7104 7205 7215 7543 7612 7648 7733 7748
7751 7813 7905 7967 8006 8028 8031 8034 8038 8067 8068 8070 8071 8081`

- **Shape: a tight ceiling near 8030 plus a spread tail.** The 12
  readings at or above 7900 have sd 51; the campaign's upper half has sd
  111. The first draft's "same two levels" left out the readings in
  between (7543, 7612, 7648, 7733, 7748, 7751, 7813) and is withdrawn.
  This matches the original 2026-08-13 #195 description.
- **The 1.36× in #195's title is max/min of this tail**, 8081/5935. 5935
  is the 2026-07-31 15:44 REGRESSION, the push-blocking event #195 is
  about. It is not an outlier to set aside. Without it the span is 1.23×.
- **The power-capped igemm is not necessary for a low reading; a
  contribution is not excluded.** Of the 25 runs that contained it, 24
  had igemm `SwPowerCap`, with conv2d high in 13 and low in 11. Only 3
  runs had no igemm (2 high, 1 low), and all 3 started within about
  65 s of a capped run at 74 °C, so they are no clean control. The first
  draft's "ruled out by the trace" is withdrawn.

## Next steps (proposed, not run; see #223)

The first draft proposed a 100 ms `nvidia-smi` logger and a permuted
config order. The review showed both are mis-sized:
- A 100 ms logger takes about one sample per ~106 ms timed window.
- Permuting the order does not separate shape from *time since the GPU
  resumed*, because the CPU-reference interval itself depends on the
  shape.

Better instruments:

1. **Time each launch inside the bench.** It currently times 100 launches
   with one `BenchTimer` (CUevent pair) and reports the mean. Record an
   event pair per launch instead, and keep the distribution. That shows whether
   a low reading is a few slow launches (a clock ramp, a throttle event)
   or a uniformly slow window.
2. **Remove the idle before timing.** Spin the GPU, or precompute the CPU
   reference outside the config loop, so the timed section starts from a
   known clock state. If the spread and tail disappear, the clock ramp is
   the mechanism.
3. **Only then a telemetry log**, at a rate that resolves the window, run
   for the whole session so both conditions are loaded equally.

## Corrections log (after review round 1 on #224)

| first draft (c6b9f40, posted to #195) | corrected |
|---|---|
| "the idle gap cannot account for the span" | the spread does not depend on process-entry state; the manipulation probably never reached the timed window (design limitation) |
| "no recorded variable separates the groups" / "host load … match" | host load is nominally associated (p = 0.011 uncorrected), in the direction opposite to contamination |
| "two clusters" / "the gate shows the same two levels" | consistent with a single mode; the gate is a tight ceiling plus a spread tail |
| "#195's 1.36× is one outlier" | 5935 *is* the push-blocking event #195 is about |
| "ruled out by the trace" (preceding igemm) | not necessary; a contribution is not excluded |
| "whatever moves it happens inside the timed window" | the cause is not located |
| "every number is regenerated by the script" | true now: gate extract committed, Part 4 added |
| "load and drift cannot be separated" | they can (r = −0.31) |
| deviations: "the rest of the plan ran as written" | three more deviations disclosed (items 3–5) |

# Session handoff

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

| #   | State    | Title                                                                                        |
|-----|----------|----------------------------------------------------------------------------------------------|
| 32  | open     | Research: polyhedral spring networks (literature scoping shipped in `polyhedral_spring_networks.md`; outside core kernel scope) |
| 103 | open     | Cross-attention regime dispatch helper: select v2 / v2_pad vs baseline by (seq_q × seq_kv) threshold |
| 104 | tracking | Fragment-shfl reduction pattern — no current target kernel                                   |
| 105 | tracking | Tracking: #96 sub-task C — non-Tensor-Core SASS hand-tunes (speculative, no target)         |

All other issues (#1 through #102 minus the four above) are closed;
see GitHub issue history.

### Highest-EV remaining code change

**#103 cross-attention regime dispatch helper.** Add at every
cross-attention call site:

```cpp
if ((size_t)seq_kv * seq_q >= 200000) launch_v2(); else launch_baseline();
```

Measured threshold: above 200 K, v2 wins 1.43–1.62×; below 200 K
(CLIP-77 regime), v2 loses 19 %. A 30-minute change, well-measured
boundary.

All other open items are research-grade or speculative.

## Latest session — documentation and structure review

Goal: lean, production-grade research project. Pristine code
quality, neutral technical documentation, no decorative prose, all
parts coherent.

Six work batches, six commits:

| Batch | Commit  | Scope                                                            |
|-------|---------|------------------------------------------------------------------|
| A     | bcdb751 | Delete 16 shipped `.github/issues/*.md` seed files + `scripts/create_issues.sh` + empty `tools/`. |
| B     | adb138c | Add `AGENTS.md` canonical agent reference, `CHANGELOG.md`, `docs/index.md`. Rewrite `CLAUDE.md` and `.github/copilot-instructions.md` as thin pointers to `AGENTS.md`. Fold `.github/SESSION_INSIGHTS_2026-05-05.md` into the CHANGELOG. |
| C     | 47c1311 | Cosmetic pass across 22 files: drop status emojis, remove "Tier N" jargon from user-facing docs, strip "this session" from tutorial chapters and `README.md`, add a framing preamble to `gpu_reflections.md` documenting the first-person voice as a deliberate experiment and the three-segment numbering scheme. Drop the stale "Reprioritization for next sprint" section from `ncu_metrics.md`. |
| D     | a4cf2f4 | Move regenerable data to top-level `data/` (`baselines.json`, `reference_baselines.json`, `sass_histogram.csv`, `register_audit.csv`). Retire `docs/kernels.md`; rename `docs/kernels_by_family.md` → `docs/inventory.md`. De-phase per-kernel READMEs (`tutorial`, `flash_attention`, `igemm`) and the troubleshooting doc; drop `Makefile` `phase1..phase5` aliases. Trim `README.md` Documentation section to a short orientation pointing at `docs/index.md`. |
| E     | 1c03879 | Collapse five identical per-family `bench_%.cu` Makefile rules into a single `$(eval)` loop over `BENCH_VARIANT_DIRS`. |
| F     | (this)  | Rewrite this `CONTINUE_HERE.md` as a concise session-handoff document. Prior 677-line session log retired; durable content folded into the sections above. |

Cumulative effect: agent-facing reference unified into `AGENTS.md`;
data and documentation cleanly separated; kernel inventory has one
canonical entry point; phase vocabulary deleted from on-disk
artifacts; structural reorganization history captured in
`CHANGELOG.md`; cosmetic noise removed without touching the
deliberate first-person experiment in `gpu_reflections.md`.

No source, kernel, baseline, or build behaviour changed. Verified
that `Rscript scripts/bench/bench_regress.R --list` still reads the
relocated baselines, and that all five collapsed bench-variant
targets resolve under `make -n`.

## Next steps

In order of expected value:

1. **Land #103 cross-attention regime dispatch helper** (30 min).
   Single concrete remaining performance change in the queue.

2. **Run `Rscript scripts/audit/sass_histogram.R` and
   `Rscript scripts/audit/reg_audit.R`** to repopulate
   `data/sass_histogram.csv` and `data/register_audit.csv` from the
   current cubin set, confirming the relocated output paths land in
   `data/` not the old `docs/` location.

3. **Run `make reproduce` end-to-end** as a smoke test of the
   restructured paths. The pre-push hook covers the bench regression
   but does not exercise `make all` from scratch.

4. **Refresh `.gitignore`**: stale `phase{1,2,4}/` entries, dead
   `tools/CuAssembler/`, missing `viz/` entries. Local
   `viz/.gitignore` already handles `dist/` and `node_modules/`, but
   the root file is documented as load-bearing in the
   2026-05-05 session insights (now in `CHANGELOG.md`).

5. **Decide on `scripts/fix_cuda_context.R`**: it is a one-shot
   codemod that has run; either delete it or move it to a
   `scripts/migrations/` subdirectory.

6. **#32 polyhedral spring networks**: literature scoping is done
   (`docs/polyhedral_spring_networks.md`). Next decision is whether
   to pursue a concrete kernel implementation or close the issue.

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

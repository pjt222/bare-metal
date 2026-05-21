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
| 122 | open     | `make clean` deletes tracked artifacts (handtuned `.cuasm`, experiment cubins/sass)          |

All other issues are closed; see GitHub issue history.

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

## Latest session — documentation review + Hugging Face publication

Goal: bring the repo to scientific-paper standard, then publish the
kernel corpus on Hugging Face. Tracked as epic #110 + four workstream
issues; executed via the `bare-metal-docs` agent team.

| WS | Issue | Commit  | Scope                                                          |
|----|-------|---------|----------------------------------------------------------------|
| 1  | #106  | ee94217 | Doc correctness: `README.md` "CUDA 12.x"→"13.2", toolchain-provenance footnotes, `verify_setup.R` driver-version capture, `Makefile` `figures` target. |
| 2  | #107  | 3c33b40 | CI: `.github/workflows/docs.yml` — markdown link-check, version-consistency (`scripts/check_versions.R`), Quarto render. |
| 3  | #108  | 2da9fbc | `README.md` restructured 367→86 lines as a paper-style abstract; showcase tables relocated to canonical homes (`inventory.md`, `AGENTS.md`, `index.md`). |
| 4  | #109  | ae5c69a | HF publish tooling: `scripts/publish_hf.R`, `hf/README.md` dataset card, `make publish-hf` re-sync target. |

Result: the kernel corpus is published as a Hugging Face **dataset**
repo — **https://huggingface.co/datasets/pjt222/ga104-cuda-kernels**
(356 entries, commit `5c3d532`). Re-sync with `make publish-hf`
(needs `HF_TOKEN` in `.env`).

Bugs found and fixed while exercising the full reproducible pipeline:

| #   | Commit  | Fix                                                              |
|-----|---------|------------------------------------------------------------------|
| 117 | 84ca4b2 | `igemm_online_quant_bf_128x256` 64 KB static smem > sm_86 48 KB limit; unioned `epilogue_tile` over the A/B double-buffer → 48 KB. |
| 116 | 0f3c198 | `verify_setup.R` nvidia-smi exit 9 — R's `LD_LIBRARY_PATH` shadowed the WSL `libnvidia-ml.so`; also renv upgraded 1.2.2→1.2.3. |
| 119 | 7194e5d | `cross_attention` missing from `BENCH_VARIANT_DIRS` — broke `make all`. |
| 120 | adc80d9 | `hf repo create` rejected `datasets/` prefix + `--type` together. |
| 121 | cd06d0a | Fast `make sass` target (cuobjdump-only, ~60 s vs ~1 h `make disasm`); `publish_hf.R --skip-build`. |

`make all` now builds the whole corpus exit 0; the publish pipeline
is reproducible end to end.

## Next steps

In order of expected value:

1. **#122 — `make clean` deletes tracked artifacts.** `make clean`'s
   broad `find -delete` removes git-tracked files (handtuned
   `.cuasm`, experiment cubins/sass). Scope it to generated output.

2. **Land #103 cross-attention regime dispatch helper** (30 min).
   Single concrete remaining performance change in the queue.

3. **Refresh `.gitignore`**: stale `phase{1,2,4}/` entries, dead
   `tools/CuAssembler/`, missing `viz/` entries.

4. **Decide on `scripts/fix_cuda_context.R`**: one-shot codemod that
   has run; delete it or move to `scripts/migrations/`.

5. **#32 polyhedral spring networks**: literature scoping is done
   (`docs/polyhedral_spring_networks.md`). Decide whether to pursue a
   kernel implementation or close the issue.

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

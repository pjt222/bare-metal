# `scripts/` — R analysis and tooling

All R scripts run under the project's `renv` library (R 4.6.0,
`renv.lock` pinned). First-time setup: `Rscript -e 'renv::restore()'`.

The few shell scripts (`*.sh`) are environment / repo-management
helpers that don't depend on R.

## Directory map

```
scripts/
├── build.R                   ── compile / disasm / roundtrip (uses local cuasmR)
├── verify_setup.R            ── environment check (CUDA, GPU, cuasmR, renv)
├── install_cuasmR.R          ── (re)install the local cuasmR R package
├── install-hooks.sh          ── install repo-local git hooks
│
├── model/                    ── analytical perf models (no GPU required)
│   ├── occupancy_calc.R      ── block params → warps/SM, bottleneck identifier
│   ├── perf_model_panel.R    ── roofline + memory ceiling for (M, N, K)
│   ├── pipeline_balance.R    ── compute/memory ratio per inner-loop tile
│   ├── analyze_smem_layout.R ── bank-conflict prediction for ldmatrix.x4
│   ├── find_optimal_smem_layout.R ── sweep (BM, BN, BK) for 2 blocks/SM
│   ├── config_optimizer.R    ── pick best launch config from sweep results
│   └── kernel_dashboard.R    ── combined dashboard of all four
│
├── cymatic/                  ── Chladni-pattern memory layout pipeline
│   ├── cymatic_mapping.R     ── (n, m) modes → permutation table → perm.bin
│   ├── cymatic_analyze.R     ── locality metric (cym vs row, per trace)
│   ├── cymatic_visualize.R   ── render PNG figures (regions, addresses, fields)
│   ├── cymatic_optimize.R    ── sweep (n, m) ∈ {2..10}×{1..6}, find per-trace best
│   ├── cymatic_optimize_summary.R ── tabulate optimize.R sweep output
│   └── cymatic_fa_alignment.R ── how cymatic regions align with FA seq tiles
│
├── bench/                    ── benchmark harnesses (need GPU)
│   ├── bench_regress.R       ── runs all benches vs data/baselines.json
│   ├── bench_reference.R     ── runs local reference benches vs data/reference_baselines.json
│   ├── compare_reference.R   ── joins project baselines to local reference baselines
│   ├── bench_flash_all.R     ── runs every FA variant in kernels/attention/flash_attention/, prints table
│   ├── bench_imma_s02.R      ── one-off: IMMA S02 vs S04 stall comparison
│   ├── bench_imma_s04.R      ── companion to s02
│   └── handtune_imma_s04.R   ── byte-patch S04 → S02 via cuasmR + run
│
├── profile/                  ── NCU profiling + measured roofline
│   ├── ncu_profile.R         ── ncu wrapper for one kernel → markdown row
│   ├── ncu_profile_all.sh    ── sweep all 10 representative kernels
│   └── roofline_measured.R   ── synthesise NCU output → roofline figure
│
└── audit/                    ── source / SASS audits (mostly cubin-walking)
    ├── sass_histogram.R      ── walk all .cubin, count opcodes by family
    ├── reg_audit.R           ── per-kernel register usage table
    ├── ldmatrix_conflicts.R  ── predict ldmatrix.x4 bank-conflict count
    ├── track_prmt_reduction.R ── trace PRMT instruction reduction across rewrites
    └── generate_readme_figures.R ── regenerate README.md headline figures
```

## Common entry points

| Goal | Command |
|---|---|
| Build a single cubin                        | `Rscript scripts/build.R compile path/to/kernel.cu` |
| Disassemble + cuasmR roundtrip a cubin      | `Rscript scripts/build.R disasm path/to/kernel.sm_86.cubin` |
| Run all benches vs baselines                | `Rscript scripts/bench/bench_regress.R` |
| Run local reference benches                 | `Rscript scripts/bench/bench_reference.R` |
| Compare project vs local references         | `Rscript scripts/bench/compare_reference.R` |
| Profile a kernel with ncu                   | `Rscript scripts/profile/ncu_profile.R --kernel path/to/bench --out results/ncu/foo.csv` |
| Sweep occupancy / SMEM layout               | `Rscript scripts/model/find_optimal_smem_layout.R` |
| Regenerate cymatic data + figures           | `Rscript scripts/cymatic/cymatic_mapping.R 2048 6 4 && Rscript scripts/cymatic/cymatic_visualize.R` |
| Regenerate README headline figures          | `Rscript scripts/audit/generate_readme_figures.R` |
| Refresh SASS histogram + register audit     | `Rscript scripts/audit/sass_histogram.R && Rscript scripts/audit/reg_audit.R` |

## Notes

- Scripts that fork CUDA processes (anything calling `nvcc`, `ptxas`,
  `cuobjdump`, `ncu`) need `LD_LIBRARY_PATH=/usr/lib/wsl/lib:...` so
  the WSL `libcuda.so` passthrough is found. `build.R` handles this
  internally; ad-hoc invocations may need it set in the calling shell.
- All file outputs (CSVs, RDS, PNG) land under `docs/figures/`,
  `results/`, or alongside their generating data
  (e.g. `kernels/memory_layout/cymatic/` for cymatic outputs).

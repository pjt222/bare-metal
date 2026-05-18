# experiments/ — Front-end and tooling experiments

Not a kernel phase. Phases 1–5 are SASS hand-edit research on
specific GPU compute kernels; this directory is a sandbox for
experiments that sit *above* SASS — alternative front-ends, codegen
backends, and developer-experience tooling.

> This directory was originally named `phase6/` and was renamed to
> `experiments/` to drop the misleading `phaseN/` prefix; the work
> here is not a numbered kernel phase. See `CHANGELOG.md`.

## Sub-experiments

| Dir | Topic | Status | Verdict |
|---|---|---|---|
| [`rust-experiments/`](rust-experiments/)               | cuda-oxide Rust→PTX vecadd spike (Obs KK)      | done | 2× SASS bloat on safety-typed kernels; pipeline portable |
| [`rust-experiments/cymatic_oxide/`](rust-experiments/cymatic_oxide/) | cuda-oxide on `gather_sum` kernel (Obs LL) | done | oxide 0.67× SASS but 0.65–0.80× runtime; nvcc unroll heuristic dominates |

## Cross-references

- [docs/gpu_reflections.md](../docs/gpu_reflections.md) — Obs KK, LL writeups
- [docs/cuasm_r.md](../docs/cuasm_r.md) — local R package for SASS hand-edits (replaces upstream CuAssembler)
- [docs/kernels.md](../docs/kernels.md) — kernel phase mapping (phase 6 row links here)

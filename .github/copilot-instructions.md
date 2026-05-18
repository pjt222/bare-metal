# Copilot instructions

The canonical agent-facing reference is **[`AGENTS.md`](../AGENTS.md)** at
the repository root. Read it first.

`AGENTS.md` documents hardware constraints (GA104, sm_86, the 50 KB
shared-memory cliff), the toolchain (CUDA 13.2, R 4.6.0, renv-pinned),
build entry points (`make reproduce`, family targets), code
conventions (`extern "C"`, `__launch_bounds__`, `__restrict__`,
size_t indices), the CUDA Driver API bench convention, correctness
tolerances, the SASS hand-edit workflow via the local `cuasmR` R
package, and the four laws of GA104.

No Copilot-specific addenda apply at this time. Suggestions should
conform to the conventions in `AGENTS.md`.

# scripts/bench/bench_meta.R -- compatibility shim (issue #134).
#
# The GPU/host state capture functions — capture_gpu_state(),
# classify_meta(), decode_throttle(), summarise_meta() — moved into the
# cuasmR package (R/cuasmR/R/bench_meta.R). This shim remains so existing
# `source("scripts/bench/bench_meta.R")` calls keep resolving them; it
# just attaches cuasmR. Slated for removal in a future release once all
# callers use library(cuasmR) directly.
suppressMessages(library(cuasmR))

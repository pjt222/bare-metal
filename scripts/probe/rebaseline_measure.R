#!/usr/bin/env Rscript
# scripts/probe/rebaseline_measure.R
#
# Measurement driver for the hgemm + igemm re-baseline
# (see docs/rebaseline_protocol.md). The 2026-05-10 baselines were
# recorded under a power-supply fault; this collects a clean
# replacement set at the sustained cold-clock — clock-locking is
# unavailable on this machine (#125).
#
# Per config: warm the GPU to its steady-state clock, then collect
# N valid samples. A sample is valid only if the bench did not crash,
# the output parsed, no disallowed throttle fired, and the SM clock
# stayed at the sustained-cap regime (>= MIN_CLK) — a lower clock
# means the GPU decayed off the cap mid-run and the number is not
# comparable to a cold-clock baseline.
#
# Reports the per-config median; never writes baselines.json — the
# medians are reviewed by hand before being committed.
#
# Run from anywhere in the repo:
#   Rscript scripts/probe/rebaseline_measure.R

suppressPackageStartupMessages({
  library(here)
  library(fs)
  library(cli)
  library(data.table)
})

# WSL CUDA libraries must be on LD_LIBRARY_PATH for nvidia-smi /
# the CUDA driver to resolve under a fresh Rscript process.
local({
  wsl_lib <- "/usr/lib/wsl/lib"
  cur <- Sys.getenv("LD_LIBRARY_PATH")
  if (dir_exists(wsl_lib) && !grepl(wsl_lib, cur, fixed = TRUE)) {
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                   paste(wsl_lib, cur, sep = ":")
                                 else wsl_lib)
  }
})

# GPU/host state capture now lives in the cuasmR package (issue #134;
# was source("scripts/bench/bench_meta.R")).
suppressMessages(library(cuasmR))

# ---- Parameters ------------------------------------------------------

params <- list(
  warmup  = 30L,    # discard runs to reach the steady-state clock
  min_clk = 1300L,  # MHz; samples below this decayed off the ~1410 cap
  max_try_factor = 4L,  # attempt cap = nvalid * this
  log_file     = here("scripts", "probe", "rebaseline_measure.log"),
  results_file = here("scripts", "probe", "rebaseline_results.rds")
)

# Configs to re-record. `nvalid` is per-config: igemm 4096³ is
# documented-bimodal (boost vs steady at the same clock — see the
# baselines.json note), so it needs more samples for a stable median.
configs <- list(
  list(id = "hgemm 2048", kernel = "kernels/gemm/hgemm/hgemm_16warp.cu",
       cfgkey = "2048_2048_2048", exe = "kernels/gemm/hgemm/bench",
       args = c("2048", "2048", "2048"), nvalid = 7L,
       match = "hgemm_16warp (128x128 2blk/SM)", label = "GFLOPS"),
  list(id = "hgemm 4096", kernel = "kernels/gemm/hgemm/hgemm_16warp.cu",
       cfgkey = "4096_4096_4096", exe = "kernels/gemm/hgemm/bench",
       args = c("4096", "4096", "4096"), nvalid = 7L,
       match = "hgemm_16warp (128x128 2blk/SM)", label = "GFLOPS"),
  list(id = "igemm 2048", kernel = "kernels/gemm/igemm/igemm_sparse_tiled.cu",
       cfgkey = "2048_2048_2048", exe = "kernels/gemm/igemm/bench_sparse",
       args = c("2048", "2048", "2048"), nvalid = 7L,
       match = "igemm_sparse_tiled", label = "dense-equiv GFLOPS"),
  list(id = "igemm 4096", kernel = "kernels/gemm/igemm/igemm_sparse_tiled.cu",
       cfgkey = "4096_4096_4096", exe = "kernels/gemm/igemm/bench_sparse",
       args = c("4096", "4096", "4096"), nvalid = 15L,
       match = "igemm_sparse_tiled", label = "dense-equiv GFLOPS")
)

# ---- Bench execution -------------------------------------------------

# run_bench() (run + GPU-state capture) now lives in cuasmR (issue #134);
# the caller still chdir's into the exe dir first.

# Throughput parsing now lives in cuasmR::parse_throughput (issue #134).
# `cfg$label` maps to the unified value_label arg; pick = "last" (the
# perf line is the last match-bearing line). The parser returns
# `$throughput` (was `$tput` in the old local parse_bench_line).

#' Classify one run against the cold-clock recording regime.
#' Returns "valid" or a rejection reason.
classify_sample <- function(run, parsed, cfg) {
  if (run$rc != 0L) return("crash")
  if (is.na(parsed$throughput)) return("parse-fail")
  meta <- classify_meta(run$pre, run$post,
                        list(require_no_throttle = TRUE,
                             allow_throttle = c("GpuIdle")))
  if (isFALSE(meta$ok)) return("throttled")
  clk <- run$post$gpu$clock_sm
  if (is.na(clk) || clk < params$min_clk)
    return(sprintf("cold-clock %d MHz", as.integer(clk)))
  "valid"
}

# ---- Per-config measurement -----------------------------------------

#' Drive the GPU to steady state with `n` discarded runs, logging the
#' clock periodically so the warmup can be audited.
warm_gpu <- function(exe_abs, args, n) {
  cli_alert_info("Warmup: {n} discarded runs")
  for (i in seq_len(n)) {
    run <- run_bench(exe_abs, args)
    if (i %% 10L == 0L || i == n)
      cli_alert("  warmup {i}/{n}: SM clock {run$post$gpu$clock_sm} MHz")
  }
}

#' Collect `cfg$nvalid` valid samples for one config. Returns a
#' data.table of valid samples (tput, ms, clk).
collect_samples <- function(cfg, exe_abs) {
  max_try <- cfg$nvalid * params$max_try_factor
  rows    <- vector("list", max_try)
  n_valid <- 0L
  attempt <- 0L
  while (n_valid < cfg$nvalid && attempt < max_try) {
    attempt <- attempt + 1L
    run     <- run_bench(exe_abs, cfg$args)
    parsed  <- parse_throughput(run$out, match = cfg$match,
                                value_label = cfg$label, pick = "last")
    verdict <- classify_sample(run, parsed, cfg)
    clk     <- run$post$gpu$clock_sm
    if (verdict == "valid") {
      n_valid <- n_valid + 1L
      rows[[n_valid]] <- data.table(
        tput = parsed$throughput, ms = parsed$ms, clk = as.integer(clk))
      cli_alert_success(paste0(
        "try {attempt}: {round(parsed$throughput, 1)} {cfg$label}  ",
        "{round(parsed$ms, 3)} ms  clk {clk} MHz  ",
        "[{n_valid}/{cfg$nvalid}]"))
    } else {
      cli_alert_warning("try {attempt}: rejected -- {verdict}")
    }
  }
  if (n_valid < cfg$nvalid)
    cli_alert_danger(
      "INCOMPLETE: {n_valid}/{cfg$nvalid} valid after {attempt} tries")
  rbindlist(rows[seq_len(n_valid)])
}

#' Measure one config end to end. chdir into the exe directory so the
#' bench finds its cubin, restoring the working directory after.
measure_config <- function(cfg) {
  cli_h2("{cfg$id}  ({cfg$cfgkey})")
  exe_abs <- path_abs(here(cfg$exe))
  if (!file_exists(exe_abs))
    cli_abort("Executable not found: {exe_abs} (run `make all`)")

  withr_dir <- getwd()
  on.exit(setwd(withr_dir), add = TRUE)
  setwd(path_dir(exe_abs))

  warm_gpu(exe_abs, cfg$args, params$warmup)
  samples <- collect_samples(cfg, exe_abs)

  if (nrow(samples) == 0L) {
    cli_alert_danger("{cfg$id}: NO valid samples -- config not measured")
    return(list(
      id = cfg$id, kernel = cfg$kernel, cfgkey = cfg$cfgkey,
      label = cfg$label, n = 0L,
      median_tput = NA_real_, median_ms = NA_real_,
      tput_lo = NA_real_, tput_hi = NA_real_,
      clk_lo = NA_integer_, clk_hi = NA_integer_))
  }

  summary <- list(
    id = cfg$id, kernel = cfg$kernel, cfgkey = cfg$cfgkey,
    label = cfg$label, n = nrow(samples),
    median_tput = stats::median(samples$tput),
    median_ms   = stats::median(samples$ms),
    tput_lo = min(samples$tput), tput_hi = max(samples$tput),
    clk_lo  = min(samples$clk),  clk_hi  = max(samples$clk))
  cli_alert_info(paste0(
    "median {round(summary$median_tput, 1)} {cfg$label} ",
    "({round(summary$tput_lo,1)}-{round(summary$tput_hi,1)}), ",
    "{round(summary$median_ms, 4)} ms, clk {summary$clk_lo}-{summary$clk_hi} MHz"))
  summary
}

# ---- Reporting -------------------------------------------------------

report_summary <- function(results) {
  cli_h1("Summary -- proposed baselines.json values")
  for (r in results) {
    cli_text(
      "{.strong {sprintf('%-12s', r$id)}} {r$cfgkey} : ",
      "{sprintf('%.4f', r$median_ms)} ms / {round(r$median_tput)} ",
      "(n={r$n}, clk {r$clk_lo}-{r$clk_hi} MHz)")
  }
}

# ---- Entry point -----------------------------------------------------

main <- function() {
  log_con <- file(params$log_file, open = "wt")
  sink(log_con, split = TRUE)
  on.exit({ sink(); close(log_con) }, add = TRUE)

  cli_h1("Re-baseline measurement -- hgemm + igemm")
  cli_text("date: {format(Sys.time(), '%Y-%m-%d %H:%M:%S')}")

  state <- capture_gpu_state()
  cli_text(paste0(
    "AC: {state$host$ac_state} | start clock {state$gpu$clock_sm} MHz | ",
    "temp {state$gpu$temp_c} C"))
  if (identical(state$host$ac_state, "battery"))
    cli_abort("On battery -- the protocol requires AC power. Aborting.")

  results <- lapply(configs, measure_config)
  report_summary(results)

  saveRDS(results, params$results_file)
  cli_alert_success("Results saved: {.path {params$results_file}}")
  cli_alert_info("Log: {.path {params$log_file}}")
}

if (sys.nframe() == 0L) main()

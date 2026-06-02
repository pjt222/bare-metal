#!/usr/bin/env Rscript
# scripts/probe/clock_lock_sweep.R
#
# Clock-lock probe for igemm 4096³ (and any other power-bound config).
#
# Context: igemm 4096³ has no fair no-throttle baseline at native boost
# — the bench averages 50 kernel launches (bench.cu:642), and an
# unknown fraction of those 50 hit the 150 W SwPowerCap mid-run. The
# pre/post GPU snapshot only sees the boundary, so the averaged number
# is bimodal (~1.9× spread) and the per-run "throttled" gate is blind
# to it.
#
# #125 was closed "clock-lock unavailable" on a WSL-side test only.
# The Windows-host `nvidia-smi.exe -lgc` DOES work. Locking the SM
# clock low enough keeps every inner iteration under 150 W, so no
# iteration throttles and the averaged number stabilises.
#
# This script does NOT set the clock — it cannot, the lock needs an
# elevated Windows shell. The operator locks the clock externally
# (`nvidia-smi -lgc X,X` in an Administrator PowerShell) and leaves it
# locked; this script measures under whatever lock is in force, records
# EVERY sample with its full throttle state (no rejection gate), and
# reports the spread so the operator can decide whether to lock lower.
#
# Run from the repo root, after the clock is locked:
#   Rscript scripts/probe/clock_lock_sweep.R [nsamples]
# Default nsamples = 12.
#
# Each run appends one labelled block to scripts/probe/clock_lock_sweep.rds
# (keyed by the observed locked clock) so a downward sweep accumulates.

suppressPackageStartupMessages({
  library(here)
  library(fs)
  library(cli)
  library(data.table)
})

# WSL CUDA libraries must be on LD_LIBRARY_PATH for nvidia-smi / the
# CUDA driver to resolve under a fresh Rscript process.
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

args_cli   <- commandArgs(trailingOnly = TRUE)
n_samples  <- if (length(args_cli) >= 1L)
                as.integer(args_cli[[1]]) else 12L
# 8 warmup invocations did not reach steady state at high locked
# clocks — the first measured sample came back slow at 1500/1605
# every time. 20 invocations (~3-4 s of kernel time) settles it; the
# first sample is still excluded from the clean stats as a belt-and-
# braces guard. Both are reported, so the artifact stays visible.
warmup_n      <- 20L
discard_first <- 1L

config <- list(
  id     = "igemm 4096",
  cfgkey = "4096_4096_4096",
  exe    = "kernels/gemm/igemm/bench_sparse",
  args   = c("4096", "4096", "4096"),
  match  = "igemm_sparse_tiled",
  label  = "dense-equiv GFLOPS")

results_file <- here("scripts", "probe", "clock_lock_sweep.rds")
log_file     <- here("scripts", "probe", "clock_lock_sweep.log")

# ---- Bench execution (shared shape with rebaseline_measure.R) --------

run_bench <- function(exe_abs, args) {
  pre  <- capture_gpu_state()
  out  <- suppressWarnings(
    system2(exe_abs, args, stdout = TRUE, stderr = TRUE))
  post <- capture_gpu_state()
  status <- attr(out, "status")
  list(out = out, pre = pre, post = post,
       rc = if (is.null(status)) 0L else as.integer(status))
}

parse_bench_line <- function(out, match, label) {
  cand <- grep(match, out, fixed = TRUE, value = TRUE)
  cand <- cand[grepl(label, cand, fixed = TRUE) &
               grepl("ms",  cand, fixed = TRUE)]
  if (length(cand) == 0L) return(list(ms = NA_real_, tput = NA_real_))
  line <- cand[[length(cand)]]
  label_rx <- gsub("([().])", "\\\\\\1", label)
  list(
    ms   = suppressWarnings(as.numeric(
             sub(".*?([0-9.]+)\\s*ms.*", "\\1", line))),
    tput = suppressWarnings(as.numeric(
             sub(sprintf(".*?([0-9.]+)\\s*%s.*", label_rx), "\\1", line)))
  )
}

# Decode the post-run throttle reasons to a single string. GpuIdle is
# benign (the GPU idled after the bench finished); anything else is a
# real throttle that fired at the boundary — and, for an averaged
# bench, a hint that inner iterations throttled too.
throttle_label <- function(post) {
  reasons <- setdiff(post$gpu$throttle, "GpuIdle")
  if (length(reasons) == 0L) "none" else paste(reasons, collapse = ",")
}

# ---- Measurement -----------------------------------------------------

main <- function() {
  log_con <- file(log_file, open = "wt")
  sink(log_con, split = TRUE)
  on.exit({ sink(); close(log_con) }, add = TRUE)

  cli_h1("Clock-lock probe -- {config$id}")
  cli_text("date: {format(Sys.time(), '%Y-%m-%d %H:%M:%S')}")

  exe_abs <- path_abs(here(config$exe))
  if (!file_exists(exe_abs))
    cli_abort("Executable not found: {exe_abs} (run `make all`)")

  state <- capture_gpu_state()
  if (identical(state$host$ac_state, "battery"))
    cli_abort("On battery -- AC power required. Aborting.")
  cli_text(paste0(
    "AC: {state$host$ac_state} | start clock {state$gpu$clock_sm} MHz | ",
    "temp {state$gpu$temp_c} C"))

  withr_dir <- getwd()
  on.exit(setwd(withr_dir), add = TRUE)
  setwd(path_dir(exe_abs))

  # Warmup to drive the GPU to whatever steady clock the lock permits.
  cli_alert_info("Warmup: {warmup_n} discarded runs")
  for (i in seq_len(warmup_n)) run_bench(exe_abs, config$args)

  post_warm <- capture_gpu_state()
  locked_clk <- post_warm$gpu$clock_sm
  cli_alert_info("Observed SM clock after warmup: {locked_clk} MHz")

  cli_h2("Collecting {n_samples} samples (no rejection gate)")
  rows <- vector("list", n_samples)
  for (i in seq_len(n_samples)) {
    run    <- run_bench(exe_abs, config$args)
    parsed <- parse_bench_line(run$out, config$match, config$label)
    thr    <- throttle_label(run$post)
    clk    <- run$post$gpu$clock_sm
    rows[[i]] <- data.table(
      sample = i, tput = parsed$tput, ms = parsed$ms,
      clk = as.integer(clk), throttle = thr, rc = run$rc)
    cli_alert(paste0(
      "sample {i}/{n_samples}: ",
      "{round(parsed$tput, 0)} {config$label}  ",
      "{round(parsed$ms, 3)} ms  clk {clk} MHz  throttle={thr}"))
  }
  dt <- rbindlist(rows)

  # ---- Report --------------------------------------------------------
  # `clean` drops throttled / crashed / unparsed samples AND the first
  # `discard_first` samples (warmup-tail artifact). The headline median
  # and spread come from `clean`; the raw count is shown too so the
  # artifact rate stays auditable.
  cli_h2("Summary -- locked-clock regime {locked_clk} MHz")
  ok    <- dt[throttle == "none" & rc == 0L & !is.na(tput)]
  clean <- ok[sample > discard_first]
  cli_text(paste0(
    "samples total {nrow(dt)}, no-throttle {nrow(ok)}, ",
    "clean (excl. first {discard_first}) {nrow(clean)}"))
  if (nrow(dt) > 0L) {
    tally <- dt[, .N, by = throttle]
    for (k in seq_len(nrow(tally)))
      cli_text("  throttle={tally$throttle[k]} : {tally$N[k]}")
  }
  if (nrow(clean) >= 1L) {
    spread <- max(clean$tput) / min(clean$tput)
    within5 <- sum(clean$tput >= 0.95 * stats::median(clean$tput) &
                   clean$tput <= 1.05 * stats::median(clean$tput))
    cli_alert_success(paste0(
      "clean median {round(stats::median(clean$tput))} {config$label}  ",
      "({round(min(clean$tput))}-{round(max(clean$tput))}, ",
      "{round(spread, 3)}x spread)  ",
      "ms median {round(stats::median(clean$ms), 4)}"))
    cli_text("  within +/-5% of median: {within5}/{nrow(clean)}")
  } else {
    cli_alert_danger("no clean samples at this clock")
  }

  # ---- Persist (accumulate across sweep points) ----------------------
  block <- list(
    locked_clk = locked_clk, date = Sys.time(),
    n_samples = n_samples, samples = dt)
  prior <- if (file_exists(results_file)) readRDS(results_file) else list()
  prior[[sprintf("clk_%d_%s", locked_clk,
                 format(Sys.time(), "%H%M%S"))]] <- block
  saveRDS(prior, results_file)
  cli_alert_success("Appended to {.path {results_file}}")
  cli_alert_info("Log: {.path {log_file}}")
}

if (sys.nframe() == 0L) main()

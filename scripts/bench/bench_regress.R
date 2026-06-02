#!/usr/bin/env Rscript
# bench_regress.R - Automated performance regression checker.
#
# Runs benchmark executables and compares against recorded baselines in
# data/baselines.json. Exits non-zero if any kernel regresses beyond tolerance.
# Mirrors bench_regress.py.
#
# Usage:
#   Rscript scripts/bench/bench_regress.R                                          # all
#   Rscript scripts/bench/bench_regress.R --kernel kernels/gemm/hgemm/hgemm_16warp.cu
#   Rscript scripts/bench/bench_regress.R --tolerance 0.15
#   Rscript scripts/bench/bench_regress.R --list

library(jsonlite)

# GPU + host state pre/post each bench, now from the cuasmR package
# (issue #134; was source("scripts/bench/bench_meta.R")). capture_gpu_state,
# classify_meta, decode_throttle, summarise_meta are exported by cuasmR.
suppressMessages(library(cuasmR))

# WSL CUDA libpath (R subprocesses can't see GPU otherwise).
.WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(.WSL_CUDA_LIB) &&
    !grepl(.WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  .cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(.cur))
                                  paste(.WSL_CUDA_LIB, .cur, sep = ":")
                                else .WSL_CUDA_LIB)
}

# Walk up from the script's directory until a repo marker is found
# (.git or renv.lock). Resilient to subdir relocation (the scripts/
# moved this script into scripts/bench/, so a fixed dirname() count no
# longer hits the repo root).
REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  start <- if (length(fa)) normalizePath(dirname(sub("^--file=", "", fa[1])))
           else            normalizePath(getwd())
  cur <- start
  repeat {
    if (file.exists(file.path(cur, ".git")) ||
        file.exists(file.path(cur, "renv.lock"))) break
    parent <- dirname(cur)
    if (parent == cur) {
      cur <- start  # never found marker; fall back
      break
    }
    cur <- parent
  }
  cur
}
BASELINES_PATH    <- file.path(REPO_ROOT, "data", "baselines.json")
DEFAULT_TOLERANCE <- 0.10

# ----------------------------------------------------------------------
# Clock-lock measurement parameters (issue #131).
#
# A baseline config carrying a `clock_lock` field (integer MHz) is a
# power-bound kernel: at native boost it throttles a varying fraction
# of its averaged launches and has no fair baseline. Its baseline was
# recorded under a host-side SM clock lock. bench_regress.R only gates
# such a config when invoked with --clock-locked <MHz> matching the
# entry; the operator is asserting they have locked the clock host-side
# (elevated Windows shell: nvidia-smi.exe -lgc <MHz>,<MHz>).
#
# Even locked, ~1/12 samples can be a power excursion, so a clock-lock
# config is measured as a median of N valid runs, never single-shot.
# ----------------------------------------------------------------------
CLOCK_LOCK_BAND_MHZ <- 30L   # observed SM clock must stay within ±this of clock_lock
CLOCK_LOCK_WARMUP   <- 20L   # discarded runs to settle the GPU at the locked clock
CLOCK_LOCK_SAMPLES  <- 5L    # valid samples required for the median
CLOCK_LOCK_MAX_TRY  <- 20L   # attempt cap before declaring INSUFFICIENT

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list(kernel = NULL, tolerance = DEFAULT_TOLERANCE, list_only = FALSE,
              clock_locked = NULL)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if      (a == "--kernel")    { out$kernel    <- argv[i+1];          i <- i + 2 }
    else if (a == "--tolerance") { out$tolerance <- as.numeric(argv[i+1]); i <- i + 2 }
    else if (a == "--list")      { out$list_only <- TRUE;               i <- i + 1 }
    else if (a == "--clock-locked") {
      # Operator asserts the SM clock is locked host-side at this MHz.
      out$clock_locked <- as.integer(round(as.numeric(argv[i+1]))); i <- i + 2
    }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: bench_regress.R [--kernel KCU] [--tolerance F] [--list]",
          "[--clock-locked MHZ]\n",
          "  --clock-locked MHZ  measure clock_lock-tagged configs; assert the\n",
          "                      SM clock is locked host-side at MHZ\n",
          "                      (elevated Windows: nvidia-smi.exe -lgc MHZ,MHZ).\n",
          sep = "")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  out
}

# ----------------------------------------------------------------------
# Find executable for a kernel .cu path. Used as fallback when the
# baseline entry doesn't carry an explicit `exe` override.
# ----------------------------------------------------------------------
find_executable <- function(kernel_path) {
  base <- tools::file_path_sans_ext(basename(kernel_path))
  parent <- dirname(kernel_path)
  candidates <- c(file.path(parent, "bench"),
                  file.path(parent, paste0("bench_", base)),
                  file.path(parent, base))
  for (c in candidates) if (file.exists(c)) return(c)
  NULL
}

# ----------------------------------------------------------------------
# Output parsing
#
# Bench stdout typically holds multiple `<X> ms ... <Y> (GFLOPS|TOPS)`
# lines, one per kernel variant. The baseline entry tells us which one
# is *this* kernel's number via three optional fields (match / section /
# value_label) passed through to cuasmR::parse_throughput. The line
# selection + number extraction live there now (issue #134); a
# characterization test in the package (test-parse_throughput.R) pins
# them to the original .pick_line/.parse_line behaviour on real output.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Benchmark runner
# ----------------------------------------------------------------------
run_benchmark <- function(exe_path, args, baseline_cfg = NULL) {
  # Benches use cuModuleLoad with a relative cubin filename, so they must
  # run from their own directory or the cubin won't be found. Resolve
  # the absolute path to the executable, then chdir into its parent for
  # the duration of the call.
  abs_exe <- normalizePath(exe_path, mustWork = TRUE)
  exe_dir <- dirname(abs_exe)
  prev_wd <- getwd()
  setwd(exe_dir)
  on.exit(setwd(prev_wd), add = TRUE)

  # Run + GPU-state capture via the cuasmR core (issue #134). run_bench
  # snapshots capture_gpu_state() pre/post (NULL on a CI box without
  # nvidia-smi) and returns the stdout+stderr line vector. The 120s
  # timeout and error->rc=1 fallback live in run_bench.
  r <- run_bench(abs_exe, args, timeout = 120)
  out <- r$out
  rc  <- r$rc
  output <- paste(out, collapse = "\n")

  metrics <- list(raw_output = output, returncode = rc,
                  meta_pre = r$pre, meta_post = r$post)

  match_str   <- if (!is.null(baseline_cfg)) baseline_cfg$match       else NULL
  section_str <- if (!is.null(baseline_cfg)) baseline_cfg$section     else NULL
  value_label <- if (!is.null(baseline_cfg)) baseline_cfg$value_label else NULL

  # Throughput parse via cuasmR::parse_throughput (issue #134). pick =
  # "first" reproduces the legacy .pick_line (first match-bearing line);
  # a characterization test (cuasmR test-parse_throughput.R) proves
  # identical ms/throughput/unit on real bench output for every config.
  parsed <- parse_throughput(out, match = match_str, section = section_str,
                             value_label = value_label, pick = "first")
  if (is.na(parsed$throughput)) {
    # Hints matched nothing; fall back to a whole-output scan (first
    # ms + GFLOPS/TOPS line anywhere), as the old .pick_line NULL path did.
    parsed <- parse_throughput(out, value_label = value_label, pick = "first")
  }
  metrics$ms           <- parsed$ms
  metrics$throughput   <- parsed$throughput
  metrics$unit         <- parsed$unit
  metrics$matched_line <- if (is.na(parsed$line)) output else parsed$line
  metrics
}

# ----------------------------------------------------------------------
# Clock-lock measurement (issue #131)
#
# Measure a power-bound, clock_lock-tagged config as a median of N
# valid runs taken under a host-side SM clock lock. A run is valid
# only if it did not crash, its output parsed, classify_meta passed
# (no disallowed throttle — this drops the occasional SwPowerCap
# excursion that still happens even under a lock), and the observed
# SM clock stayed inside the locked band [clock_lock ± BAND]. The band
# check is two-sided: a clock far *above* clock_lock means the operator
# passed --clock-locked but never actually locked the GPU.
#
# Returns either
#   list(status = "ok", current = <metrics list for check_regression>)
# or
#   list(status = "insufficient", msg = "...")
# ----------------------------------------------------------------------
measure_clock_locked <- function(exe, cfg_args, baseline_cfg, clock_lock,
                                 valid_when) {
  lo <- clock_lock - CLOCK_LOCK_BAND_MHZ
  hi <- clock_lock + CLOCK_LOCK_BAND_MHZ

  # Warmup: settle the GPU at the locked clock; results discarded.
  for (i in seq_len(CLOCK_LOCK_WARMUP)) {
    run_benchmark(exe, cfg_args, baseline_cfg = baseline_cfg)
  }

  # Collect N valid samples. run_benchmark produces a full metrics list
  # (run+capture+parse); cuasmR::validate_sample is the per-sample verdict
  # (rc / parse / classify_meta(valid_when) / two-sided locked band). The
  # loop returns FULL metrics lists so the representative sample below can
  # carry meta_pre/meta_post/matched_line/unit forward (issue #134).
  res <- collect_valid_samples(
    sample_fn = function() run_benchmark(exe, cfg_args, baseline_cfg = baseline_cfg),
    validate_fn = function(m) validate_sample(
      m$returncode, m$throughput, m$meta_pre, m$meta_post,
      valid_when = valid_when, clock_band = c(lo, hi)),
    n_valid = CLOCK_LOCK_SAMPLES, max_attempts = CLOCK_LOCK_MAX_TRY)

  if (!res$complete) {
    return(list(status = "insufficient",
                msg = sprintf(
                  "INSUFFICIENT (%d/%d valid in %d tries; rejects: %s)",
                  length(res$samples), CLOCK_LOCK_SAMPLES, res$attempts,
                  paste(utils::head(res$rejected, 6L), collapse = ", "))))
  }

  samples <- res$samples
  tputs <- vapply(samples, function(s) s$throughput, numeric(1))
  mss   <- vapply(samples,
                  function(s) if (is.null(s$ms)) NA_real_ else s$ms,
                  numeric(1))
  med <- report_median_metrics(tputs, mss)
  # Representative sample (closest to the median) carries meta + unit
  # + matched_line forward; throughput/ms are overwritten with medians.
  current <- samples[[which.min(abs(tputs - med$median_throughput))]]
  current$throughput <- med$median_throughput
  current$ms         <- med$median_ms
  list(status = "ok", current = current)
}

# ----------------------------------------------------------------------
# Regression decision
# ----------------------------------------------------------------------
check_regression <- function(current, baseline, tolerance,
                             default_valid_when = list()) {
  if (!is.null(current$returncode) && current$returncode != 0L) {
    return(list(is_reg = TRUE,
                msg = sprintf("CRASH (exit=%d)", current$returncode)))
  }

  # Refuse to compare if the GPU was in an unfair state
  # during the run (thermal throttle, sw power cap, etc). Reported as
  # SKIPPED, not REGRESSION — the measurement isn't comparable to
  # baseline regardless of what the number says.
  if (exists("classify_meta", mode = "function") &&
      !is.null(current$meta_pre) && !is.null(current$meta_post)) {
    valid_when <- if (!is.null(baseline$valid_when)) baseline$valid_when
                  else default_valid_when
    cls <- classify_meta(current$meta_pre, current$meta_post, valid_when)
    if (isFALSE(cls$ok)) {
      return(list(is_reg = FALSE, skipped = TRUE,
                  msg = sprintf("SKIPPED (%s) [%s]",
                                paste(cls$reasons, collapse = "; "),
                                cls$summary)))
    }
  }

  unit <- if (!is.null(current$unit)) current$unit else "GFLOPS"
  baseline_val <- baseline[[tolower(unit)]]
  if (is.null(baseline_val)) baseline_val <- baseline$gflops
  if (is.null(baseline_val)) baseline_val <- baseline$tops
  if (is.null(baseline_val)) baseline_val <- 0
  current_val <- if (!is.null(current$throughput)) current$throughput else 0

  if (baseline_val == 0 || current_val == 0) {
    return(list(is_reg = TRUE,
                msg = sprintf("NO_DATA (baseline=%g, current=%g)",
                              baseline_val, current_val)))
  }
  ratio <- current_val / baseline_val
  if (ratio < (1.0 - tolerance)) {
    list(is_reg = TRUE,
         msg = sprintf("REGRESSION %.1f%% of baseline (%.0f vs %.0f %s)",
                       ratio * 100, current_val, baseline_val, unit))
  } else if (ratio > (1.0 + tolerance)) {
    list(is_reg = FALSE,
         msg = sprintf("IMPROVED %.1f%% of baseline (%.0f vs %.0f %s)",
                       ratio * 100, current_val, baseline_val, unit))
  } else {
    list(is_reg = FALSE,
         msg = sprintf("OK %.1f%% of baseline (%.0f vs %.0f %s)",
                       ratio * 100, current_val, baseline_val, unit))
  }
}

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if (!file.exists(BASELINES_PATH)) {
    cat(sprintf("ERROR: Baselines file not found: %s\n", BASELINES_PATH))
    cat("Run benchmarks manually and record results to data/baselines.json\n")
    quit(status = 1)
  }
  baselines <- jsonlite::fromJSON(BASELINES_PATH, simplifyVector = FALSE)

  if (args$list_only) {
    cat(sprintf("Baselines recorded: %s\n",
                if (!is.null(baselines$recorded_date)) baselines$recorded_date else "unknown"))
    cat(sprintf("Platform: %s\n",
                if (!is.null(baselines$platform)) baselines$platform else "unknown"))
    for (kernel in names(baselines$kernels)) {
      cat(sprintf("\n%s\n", kernel))
      entry <- baselines$kernels[[kernel]]
      # Skip kernel-level metadata keys; iterate config keys only.
      cfg_names <- setdiff(names(entry), c("exe"))
      for (cfg in cfg_names) {
        d <- entry[[cfg]]
        unit <- if (!is.null(d$gflops)) "GFLOPS" else "TOPS"
        val  <- if (unit == "GFLOPS") d$gflops else d$tops
        lock_note <- if (!is.null(d$clock_lock))
                       sprintf("  [clock_lock %d MHz]", as.integer(d$clock_lock))
                     else ""
        cat(sprintf("  %s: %s ms, %s %s%s\n",
                    cfg,
                    if (!is.null(d$ms)) d$ms else "?",
                    if (!is.null(val)) val else "?",
                    unit, lock_note))
      }
    }
    quit(status = 0)
  }

  kernels <- baselines$kernels
  if (!is.null(args$kernel)) {
    if (is.null(kernels[[args$kernel]])) {
      cat(sprintf("ERROR: Kernel '%s' not found in baselines\n", args$kernel))
      quit(status = 1)
    }
    kernels <- kernels[args$kernel]
  }

  cat(strrep("=", 70), "\n")
  cat("  Performance Regression Check\n")
  cat(sprintf("  Tolerance: %.0f%%\n", args$tolerance * 100))
  cat(sprintf("  Baselines: %s\n",
              if (!is.null(baselines$recorded_date)) baselines$recorded_date else "unknown"))
  cat(strrep("=", 70), "\n")

  regressions <- 0L; improvements <- 0L; skipped <- 0L; total <- 0L

  # Reserved keys at the kernel-entry level that are not config names.
  RESERVED_KEYS <- c("exe")

  # Print one-line GPU state header so the user can see
  # whether the run started under unfair conditions.
  if (exists("capture_gpu_state", mode = "function")) {
    .pre_session <- capture_gpu_state()
    if (!is.null(.pre_session)) {
      cat(sprintf("  GPU state: %s\n",
                  summarise_meta(.pre_session, .pre_session)))
      cat(strrep("=", 70), "\n")
    }
  }

  # Project-wide default valid_when (e.g. require no throttle).
  # Per-kernel valid_when overrides this; absent both, classify_meta
  # uses its own internal defaults.
  .default_vw <- if (!is.null(baselines$default_valid_when))
                   baselines$default_valid_when else list()

  for (kernel_path in names(kernels)) {
    entry <- kernels[[kernel_path]]
    # `exe` override from baselines schema: use if present, else heuristic.
    exe <- if (!is.null(entry$exe)) entry$exe else find_executable(kernel_path)
    if (is.null(exe) || !file.exists(exe)) {
      cat(sprintf("\n%s\n  SKIP -- executable not found (try: make benches)\n",
                  kernel_path))
      next
    }
    cfg_names <- setdiff(names(entry), RESERVED_KEYS)
    for (cfg in cfg_names) {
      total <- total + 1L
      cfg_args <- strsplit(cfg, "_", fixed = TRUE)[[1]]
      baseline_cfg <- entry[[cfg]]

      # Clock-lock dispatch (#131): a config carrying a `clock_lock`
      # field is power-bound — fair only under a matching host-side
      # SM clock lock. Gated solely when --clock-locked matches;
      # SKIPPED otherwise (the pre-push hook never locks, by design).
      cl <- if (!is.null(baseline_cfg$clock_lock))
              as.integer(round(as.numeric(baseline_cfg$clock_lock))) else NULL
      if (!is.null(cl)) {
        if (is.null(args$clock_locked)) {
          cat(sprintf(paste0("\n%s [%s]\n  SKIPPED (clock_lock %d MHz; ",
                             "rerun with --clock-locked %d after a ",
                             "host-side lock)\n"),
                      kernel_path, cfg, cl, cl))
          skipped <- skipped + 1L
          next
        }
        if (!identical(args$clock_locked, cl)) {
          cat(sprintf(paste0("\n%s [%s]\n  SKIPPED (--clock-locked %d ",
                             "!= entry clock_lock %d)\n"),
                      kernel_path, cfg, args$clock_locked, cl))
          skipped <- skipped + 1L
          next
        }
        vw <- if (!is.null(baseline_cfg$valid_when)) baseline_cfg$valid_when
              else .default_vw
        ml <- measure_clock_locked(exe, cfg_args, baseline_cfg, cl, vw)
        if (identical(ml$status, "insufficient")) {
          cat(sprintf("\n%s [%s]\n  SKIPPED (%s)\n",
                      kernel_path, cfg, ml$msg))
          skipped <- skipped + 1L
          next
        }
        eff_tol <- if (!is.null(baseline_cfg$tolerance))
                     as.numeric(baseline_cfg$tolerance)
                   else args$tolerance
        verdict <- check_regression(ml$current, baseline_cfg, eff_tol,
                                    default_valid_when = .default_vw)
        cat(sprintf("\n%s [%s] (clock-locked %d MHz, median of %d)\n  %s\n",
                    kernel_path, cfg, cl, CLOCK_LOCK_SAMPLES, verdict$msg))
        if (isTRUE(verdict$skipped))      skipped <- skipped + 1L
        else if (verdict$is_reg)          regressions <- regressions + 1L
        else if (grepl("IMPROVED", verdict$msg, fixed = TRUE))
                                          improvements <- improvements + 1L
        next
      }

      current <- run_benchmark(exe, cfg_args, baseline_cfg = baseline_cfg)
      # Per-config tolerance override: some
      # kernels are intrinsically noisy on this hardware (bimodal
      # boost-state behavior) and need a wider tolerance band.
      eff_tol <- if (!is.null(baseline_cfg$tolerance))
                   as.numeric(baseline_cfg$tolerance)
                 else args$tolerance
      verdict <- check_regression(current, baseline_cfg, eff_tol,
                                  default_valid_when = .default_vw)

      cat(sprintf("\n%s [%s]\n  %s\n", kernel_path, cfg, verdict$msg))
      if (isTRUE(verdict$skipped)) skipped <- skipped + 1L
      else if (verdict$is_reg)     regressions <- regressions + 1L
      else if (grepl("IMPROVED", verdict$msg, fixed = TRUE)) improvements <- improvements + 1L
    }
  }

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("  Total: %d | Regressions: %d | Improvements: %d | Skipped: %d\n",
              total, regressions, improvements, skipped))
  if (regressions > 0L) {
    cat(sprintf("  RESULT: FAILED -- %d regression(s) detected\n", regressions))
    quit(status = 1)
  } else {
    cat("  RESULT: PASSED -- all benchmarks within tolerance\n")
    quit(status = 0)
  }
}

if (sys.nframe() == 0L) main()

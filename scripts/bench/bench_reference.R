#!/usr/bin/env Rscript
# bench_reference.R - Validate locally measured reference-library baselines.
#
# Mirrors scripts/bench/bench_regress.R but targets data/reference_baselines.json
# instead of project-kernel baselines. The intent is to keep local production-
# library references reproducible and fair-run-gated on the same machine.

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
source(file.path(script_dir, "bench_regress.R"))

REFERENCE_BASELINES_PATH <- file.path(REPO_ROOT, "data", "reference_baselines.json")

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if (!file.exists(REFERENCE_BASELINES_PATH)) {
    cat(sprintf("ERROR: Reference baselines file not found: %s\n", REFERENCE_BASELINES_PATH))
    cat("Run the local reference benches and record results to data/reference_baselines.json\n")
    quit(status = 1)
  }
  baselines <- jsonlite::fromJSON(REFERENCE_BASELINES_PATH, simplifyVector = FALSE)

  if (args$list_only) {
    cat(sprintf("Local reference baselines recorded: %s\n",
                if (!is.null(baselines$recorded_date)) baselines$recorded_date else "unknown"))
    cat(sprintf("Platform: %s\n",
                if (!is.null(baselines$platform)) baselines$platform else "unknown"))
    for (kernel in names(baselines$kernels)) {
      cat(sprintf("\n%s\n", kernel))
      entry <- baselines$kernels[[kernel]]
      cfg_names <- setdiff(names(entry), c("exe", "library"))
      for (cfg in cfg_names) {
        d <- entry[[cfg]]
        unit <- if (!is.null(d$gflops)) "GFLOPS" else "TOPS"
        val  <- if (unit == "GFLOPS") d$gflops else d$tops
        cat(sprintf("  %s: %s ms, %s %s\n",
                    cfg,
                    if (!is.null(d$ms)) d$ms else "?",
                    if (!is.null(val)) val else "?",
                    unit))
      }
    }
    quit(status = 0)
  }

  kernels <- baselines$kernels
  if (!is.null(args$kernel)) {
    if (is.null(kernels[[args$kernel]])) {
      cat(sprintf("ERROR: Reference kernel '%s' not found in data/reference_baselines.json\n",
                  args$kernel))
      quit(status = 1)
    }
    kernels <- kernels[args$kernel]
  }

  cat(strrep("=", 70), "\n")
  cat("  Local Reference Baseline Check\n")
  cat(sprintf("  Tolerance: %.0f%%\n", args$tolerance * 100))
  cat(sprintf("  Baselines: %s\n",
              if (!is.null(baselines$recorded_date)) baselines$recorded_date else "unknown"))
  cat(strrep("=", 70), "\n")

  regressions <- 0L; improvements <- 0L; skipped <- 0L; total <- 0L
  RESERVED_KEYS <- c("exe", "library")

  if (exists("capture_gpu_state", mode = "function")) {
    .pre_session <- capture_gpu_state()
    if (!is.null(.pre_session)) {
      cat(sprintf("  GPU state: %s\n", summarise_meta(.pre_session, .pre_session)))
      cat(strrep("=", 70), "\n")
    }
  }

  .default_vw <- if (!is.null(baselines$default_valid_when))
                   baselines$default_valid_when else list()

  for (kernel_path in names(kernels)) {
    entry <- kernels[[kernel_path]]
    exe <- if (!is.null(entry$exe)) entry$exe else find_executable(kernel_path)
    if (is.null(exe) || !file.exists(exe)) {
      cat(sprintf("\n%s\n  SKIP -- executable not found (try: make reference)\n", kernel_path))
      next
    }
    cfg_names <- setdiff(names(entry), RESERVED_KEYS)
    for (cfg in cfg_names) {
      total <- total + 1L
      cfg_args <- strsplit(cfg, "_", fixed = TRUE)[[1]]
      baseline_cfg <- entry[[cfg]]
      current <- run_benchmark(exe, cfg_args, baseline_cfg = baseline_cfg)
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
    cat(sprintf("  RESULT: FAILED -- %d reference regression(s) detected\n", regressions))
    quit(status = 1)
  }
  cat("  RESULT: PASSED -- all local reference baselines within tolerance\n")
  quit(status = 0)
}

if (sys.nframe() == 0L) main()

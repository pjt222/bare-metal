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

# Regression decision now lives in cuasmR::check_regression (issue #134):
# CRASH / SKIPPED (unfair GPU state) / NO_DATA / REGRESSION / OK / IMPROVED.

# ----------------------------------------------------------------------
# Run verdict (issue #176)
#
# The verdict used to branch on `regressions > 0L` alone, so a run in which
# every config SKIPPED printed
#
#   RESULT: PASSED -- all benchmarks within tolerance
#
# and exited 0 having compared nothing against anything. Observed on five
# consecutive real pushes at 7 of 7 skipped: #156 put three configs behind a
# host-side clock lock that the pre-push hook never applies, and throttle-skip
# took the remaining four. "No regressions were found" and "no measurement was
# taken" are the same number, and only one of them is good news.
#
# Three outcomes, three exit codes:
#
#   0  PASSED        at least one config measured, none of them regressed
#   1  FAILED        at least one measured regression
#   2  INCONCLUSIVE  nothing was measured; this gate certifies nothing
#
# A measured regression outranks the empty-run case: if something was measured
# and it regressed, that is real information, however many of its neighbours
# skipped.
#
# INCONCLUSIVE is a distinct code rather than a failure because the *caller*
# owns the policy, and the three callers want different things:
#
#   .githooks/pre-push            warns and allows the push (deliberate; see
#                                 the hook step table in AGENTS.md). Blocking
#                                 would reject most pushes on this laptop, and
#                                 its escape hatch -- git push --no-verify --
#                                 switches off all five hook steps including
#                                 the GPU-free R gate (#163).
#   make bench                    same policy as the hook: warn, exit 0.
#   scripts/probe/run_locked_eval.ps1
#                                 propagates it verbatim, which is right: a
#                                 deliberate locked evaluation that measured
#                                 nothing must not report success. That script
#                                 had to be repaired to do so -- PowerShell 7
#                                 was throwing on any non-zero native exit and
#                                 skipping its own capture, so it reported 1 for
#                                 both outcomes and wrote no record at all.
#
# `measured` counts configs that reached a verdict other than SKIPPED. CRASH
# and NO_DATA count as measured and also land in `regressions`, so they report
# FAILED -- they can never make an empty run look green.
# ----------------------------------------------------------------------
summarise_verdict <- function(total, measured, regressions, skipped) {
  if (regressions > 0L) {
    return(list(status = "FAILED", exit = 1L,
                msg = sprintf(
                  "FAILED -- %d regression(s) detected (%d of %d config(s) measured)",
                  regressions, measured, total)))
  }
  if (measured < 1L) {
    return(list(status = "INCONCLUSIVE", exit = 2L,
                msg = sprintf(
                  paste0("INCONCLUSIVE -- 0 of %d config(s) measured (%d skipped); ",
                         "nothing was verified"),
                  total, skipped)))
  }
  list(status = "PASSED", exit = 0L,
       msg = sprintf(
         "PASSED -- %d of %d config(s) measured, all within tolerance (%d skipped)",
         measured, total, skipped))
}

# ----------------------------------------------------------------------
# Run record (issue #186)
#
# The gate used to print its verdicts and keep nothing. On 2026-07-31 a push was
# rejected by a measured regression and *which config regressed* was
# unrecoverable: stdout was the only record, the terminal scrollback was the only
# copy, and a re-run ten minutes later passed. The finding was gone.
#
# Of the five pre-push steps this is the only one whose result cannot be
# reproduced on demand -- check_links, renv_check and `make test-r` all re-run
# identically, and `make test` does not block -- so it is the one that has to
# write itself down. A blocking check whose evidence lives in a terminal is one
# the operator is structurally tempted to `--no-verify` past, and that switches
# off all five steps.
#
# One append-only JSONL under REPO_ROOT/results/bench_regress/. Rows are appended
# as each config is decided rather than in one block at the end, so a run killed
# mid-flight -- the CUDA-wedge case -- still leaves everything it had measured.
# cuasmR::append_jsonl_row is the same crash-safe writer the grid store uses:
# atomic at line boundaries, so a hard kill can only truncate the final line.
#
# Recording must never change the verdict. Every write is wrapped: a read-only
# checkout, a full disk or an undeletable directory downgrades to a warning on
# stderr, and the exit code is still decided by the counters alone.
# ----------------------------------------------------------------------
GATE_RECORD_PATH <- file.path(REPO_ROOT, "results", "bench_regress",
                              "gate_runs.jsonl")

utc_stamp <- function(fmt = "%Y-%m-%dT%H:%M:%SZ") {
  format(Sys.time(), fmt, tz = "UTC")
}

# Compact GPU/host digest for one measured config. The full capture_gpu_state()
# snapshots are nested and repetitive; these are the fields that answer "was the
# machine in a fit state to be measuring?" the morning after.
meta_digest <- function(current) {
  if (is.null(current) || is.null(current$meta_post)) return(NULL)
  gpu <- current$meta_post$gpu
  list(
    summary  = tryCatch(summarise_meta(current$meta_pre, current$meta_post),
                        error = function(e) NA_character_),
    clock_sm = gpu$clock_sm, clock_mem = gpu$clock_mem,
    temp_c   = gpu$temp_c,   power_w   = gpu$power_w,
    pstate   = gpu$pstate,
    # as.list() keeps `throttle` a JSON ARRAY at every length. toJSON's
    # auto_unbox would otherwise emit a bare string for one reason and an array
    # for two, so a consumer that indexes [0] silently reads the first
    # CHARACTER of "SwPowerCap" on exactly the single-reason runs that are the
    # common case.
    throttle = as.list(if (length(gpu$throttle)) gpu$throttle else character(0)),
    ac_state = current$meta_post$host$ac_state)
}

# The leading token of a verdict message is its classification: OK, IMPROVED,
# REGRESSION, SKIPPED, CRASH, NO_DATA.
verdict_word <- function(msg) {
  w <- regmatches(msg, regexpr("^[A-Z_]+", msg))
  if (length(w)) w else NA_character_
}

record_row_raw <- function(row, path = GATE_RECORD_PATH) {
  tryCatch({
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    append_jsonl_row(path, row)
    TRUE
  }, error = function(e) {
    message(sprintf("WARNING: could not write the run record to %s: %s",
                    path, conditionMessage(e)))
    FALSE
  })
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
  # `measured` is the count that decides PASSED vs INCONCLUSIVE (#176).
  measured <- 0L

  # Reserved keys at the kernel-entry level that are not config names.
  RESERVED_KEYS <- c("exe")

  # Fold one config's verdict into the run counters. `<<-` targets main()'s
  # locals. The two call sites below (clock-locked and direct) each carried
  # their own copy of this ladder, so #176 would have had to add `measured`
  # in two places and every later bucket in two more.
  tally <- function(verdict) {
    if (isTRUE(verdict$skipped)) {
      skipped <<- skipped + 1L
      return(invisible(NULL))
    }
    measured <<- measured + 1L
    if (verdict$is_reg) {
      regressions <<- regressions + 1L
    } else if (grepl("IMPROVED", verdict$msg, fixed = TRUE)) {
      improvements <<- improvements + 1L
    }
    invisible(NULL)
  }

  RUN_ID <- utc_stamp("%Y%m%dT%H%M%OS3Z")
  recorded_ok <- TRUE   # false once any write has failed
  records_written <- 0L
  records_attempted <- 0L
  failed <- list()      # configs behind a FAILED verdict, recapped before RESULT

  # Count every attempt and every success, not just a boolean. Losing one row
  # out of eight and losing all eight are different situations for whoever
  # reads this afterwards, and one flag cannot tell them apart.
  record_row <- function(row) {
    records_attempted <<- records_attempted + 1L
    ok <- record_row_raw(row)
    if (ok) records_written <<- records_written + 1L else recorded_ok <<- FALSE
    ok
  }

  # ONE funnel for every per-config outcome. Printing, counting and recording
  # used to be three independent decisions taken at six sites; #176 had to
  # change the counting at two of them and could have missed the other four,
  # and #186 would have had to add recording at all six. Anything that reaches
  # a verdict goes through here, so a new outcome cannot print without being
  # counted, or be counted without being recorded.
  emit <- function(kernel_path, cfg, msg, verdict = NULL, current = NULL,
                   header = "", extra = list()) {
    cat(sprintf("\n%s [%s]%s\n  %s\n", kernel_path, cfg, header, msg))
    # A site with no verdict object did not measure the config -- that is what
    # makes it a skip -- so synthesise the shape check_regression returns.
    v <- if (is.null(verdict)) list(is_reg = FALSE, skipped = TRUE, msg = msg)
         else verdict
    tally(v)
    if (isTRUE(v$is_reg)) {
      failed[[length(failed) + 1L]] <<-
        list(kernel = kernel_path, config = cfg, msg = msg)
    }
    record_row(c(
      list(type = "config", run_id = RUN_ID, recorded_at = utc_stamp(),
           kernel = kernel_path, config = cfg,
           verdict = verdict_word(msg), msg = msg,
           measured = !isTRUE(v$skipped)),
      extra,
      list(throughput = if (!is.null(current)) current$throughput else NULL,
           unit       = if (!is.null(current)) current$unit else NULL,
           returncode = if (!is.null(current)) current$returncode else NULL,
           meta       = meta_digest(current))))
    invisible(NULL)
  }

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
    cfg_names <- setdiff(names(entry), RESERVED_KEYS)
    # `exe` override from baselines schema: use if present, else heuristic.
    exe <- if (!is.null(entry$exe)) entry$exe else find_executable(kernel_path)
    if (is.null(exe) || !file.exists(exe)) {
      # Count the configs that cannot run (#176). This branch used to `next`
      # without touching a counter, so an unbuilt corpus reported `Total: 0`
      # -- the denominator disappeared along with the measurement, and the
      # run still exited 0. Counting them keeps `total` the number of configs
      # in baselines.json rather than the number we happened to reach.
      # Reported per config rather than once per kernel (#186) so the screen
      # and the record agree on what was not measured.
      for (cfg in cfg_names) {
        total <- total + 1L
        emit(kernel_path, cfg,
             "SKIPPED -- executable not found (try: make benches)",
             extra = list(exe = if (is.null(exe)) NA_character_ else exe))
      }
      next
    }
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
      # Baseline fields recorded verbatim rather than re-resolved here: which
      # one applies is check_regression's decision, and duplicating that choice
      # is how two copies of a rule drift apart.
      cfg_extra <- list(baseline_gflops = baseline_cfg$gflops,
                        baseline_tops   = baseline_cfg$tops,
                        clock_lock      = cl,
                        clock_locked_arg = args$clock_locked)

      if (!is.null(cl)) {
        if (is.null(args$clock_locked)) {
          emit(kernel_path, cfg,
               sprintf(paste0("SKIPPED (clock_lock %d MHz; rerun with ",
                              "--clock-locked %d after a host-side lock)"),
                       cl, cl),
               extra = cfg_extra)
          next
        }
        if (!identical(args$clock_locked, cl)) {
          emit(kernel_path, cfg,
               sprintf("SKIPPED (--clock-locked %d != entry clock_lock %d)",
                       args$clock_locked, cl),
               extra = cfg_extra)
          next
        }
        vw <- if (!is.null(baseline_cfg$valid_when)) baseline_cfg$valid_when
              else .default_vw
        ml <- measure_clock_locked(exe, cfg_args, baseline_cfg, cl, vw)
        if (identical(ml$status, "insufficient")) {
          emit(kernel_path, cfg, sprintf("SKIPPED (%s)", ml$msg),
               extra = cfg_extra)
          next
        }
        eff_tol <- if (!is.null(baseline_cfg$tolerance))
                     as.numeric(baseline_cfg$tolerance)
                   else args$tolerance
        verdict <- check_regression(ml$current, baseline_cfg, eff_tol,
                                    default_valid_when = .default_vw)
        emit(kernel_path, cfg, verdict$msg, verdict = verdict,
             current = ml$current,
             header = sprintf(" (clock-locked %d MHz, median of %d)",
                              cl, CLOCK_LOCK_SAMPLES),
             extra = c(cfg_extra, list(tolerance = eff_tol,
                                       samples = CLOCK_LOCK_SAMPLES)))
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

      emit(kernel_path, cfg, verdict$msg, verdict = verdict, current = current,
           extra = c(cfg_extra, list(tolerance = eff_tol)))
    }
  }

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf(
    "  Total: %d | Measured: %d | Regressions: %d | Improvements: %d | Skipped: %d\n",
    total, measured, regressions, improvements, skipped))

  v <- summarise_verdict(total, measured, regressions, skipped)
  config_rows_written   <- records_written
  config_rows_attempted <- records_attempted

  # Recap what failed, immediately above the verdict (#186). The per-config
  # lines scroll away behind a CUDA build, and the hook's "investigate with
  # --kernel <kernel_path>" advice needs a kernel path the operator can still
  # see. CRASH and NO_DATA land here too: they are why the run failed.
  if (length(failed)) {
    cat("\n  Configs behind this verdict:\n")
    for (f in failed) {
      cat(sprintf("    %s [%s]\n      %s\n", f$kernel, f$config, f$msg))
    }
  }

  # Capture this write's result like every other one. Discarding it made the
  # summary row -- the only carrier of the verdict, the exit code and the
  # failed-config recap, i.e. exactly the evidence #186 exists to preserve --
  # the one row whose loss the "Record:" line below could not report.
  record_row(list(
    type = "run_summary", run_id = RUN_ID, recorded_at = utc_stamp(),
    verdict = v$status, exit = v$exit, msg = v$msg,
    total = total, measured = measured, regressions = regressions,
    improvements = improvements, skipped = skipped,
    tolerance = args$tolerance, clock_locked = args$clock_locked,
    kernel_filter = args$kernel,
    baselines_recorded_date = baselines$recorded_date,
    failed = lapply(failed, function(f) list(kernel = f$kernel,
                                             config = f$config, msg = f$msg)),
    git_head = tryCatch(
      # Which tree produced these numbers. Without it a store spanning weeks
      # can say a config regressed but not against what. Best-effort: a
      # detached worktree or a missing git is not worth failing a bench run
      # over.
      trimws(system2("git", c("-C", shQuote(REPO_ROOT), "rev-parse", "--short",
                              "HEAD"), stdout = TRUE, stderr = FALSE)[1]),
      error = function(e) NA_character_),
    # Snapshotted before this row is attempted, and named for what they
    # count: a summary row reporting "2 of 3 written" while being the third
    # would read as a failure on every healthy run.
    config_rows_written = config_rows_written,
    config_rows_attempted = config_rows_attempted))

  # Say what was actually written, in every outcome. A path the operator only
  # learns about when things go wrong is one they have to look up exactly when
  # they are least inclined to -- and "NOT WRITTEN" was previously printed
  # whenever ANY row failed, which called a run with seven good rows and one
  # bad one evidence-free.
  if (recorded_ok) {
    cat(sprintf("\n  Record: %s\n", GATE_RECORD_PATH))
  } else if (records_written > 0L) {
    cat(sprintf(paste0("\n  Record: PARTIAL -- %d of %d rows written to %s\n",
                       "  (see the warnings above; some of this run is missing)\n"),
                records_written, records_attempted, GATE_RECORD_PATH))
  } else {
    cat("\n  Record: NOT WRITTEN (see the warning above) -- this run left no",
        "durable evidence\n")
  }

  cat(sprintf("  RESULT: %s\n", v$msg))
  quit(status = v$exit)
}

if (sys.nframe() == 0L) main()

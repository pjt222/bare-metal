#!/usr/bin/env Rscript
# bench_regress.R - Automated performance regression checker.
#
# Runs benchmark executables and compares against recorded baselines in
# docs/baselines.json. Exits non-zero if any kernel regresses beyond tolerance.
# Mirrors bench_regress.py.
#
# Usage:
#   Rscript scripts/bench/bench_regress.R                                          # all
#   Rscript scripts/bench/bench_regress.R --kernel phase2/hgemm/hgemm_16warp.cu
#   Rscript scripts/bench/bench_regress.R --tolerance 0.15
#   Rscript scripts/bench/bench_regress.R --list

library(jsonlite)

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
# (.git or renv.lock). Resilient to subdir relocation (audit Tier 5
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
BASELINES_PATH    <- file.path(REPO_ROOT, "docs", "baselines.json")
DEFAULT_TOLERANCE <- 0.10

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list(kernel = NULL, tolerance = DEFAULT_TOLERANCE, list_only = FALSE)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if      (a == "--kernel")    { out$kernel    <- argv[i+1];          i <- i + 2 }
    else if (a == "--tolerance") { out$tolerance <- as.numeric(argv[i+1]); i <- i + 2 }
    else if (a == "--list")      { out$list_only <- TRUE;               i <- i + 1 }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: bench_regress.R [--kernel KCU] [--tolerance F] [--list]\n")
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
# is *this* kernel's number via three optional fields:
#
#   match        substring of the line containing the kernel's label
#   section      substring of an earlier section header (only consider
#                lines after this header until the next header)
#   value_label  text immediately following the throughput number on
#                the matched line; required when the line has multiple
#                numbers (e.g. 'eff GFLOPS' vs 'dense-equiv GFLOPS')
#
# Without `match`, falls back to first-ms-GFLOPS-line legacy behavior.
# ----------------------------------------------------------------------

# Heuristic: a section header is a line whose stripped form starts
# with `===` or `---` or contains '---' immediately after the trim.
# Used by the `section` filter to bracket the search.
.is_section_header <- function(line) {
  s <- trimws(line)
  grepl("^(===|---)", s, perl = TRUE)
}

# Filter raw output (a character vector of lines) to the lines inside
# the requested section. If section_match is NULL, returns lines as-is.
.filter_section <- function(lines, section_match) {
  if (is.null(section_match) || !nzchar(section_match)) return(lines)
  in_section <- FALSE
  keep <- logical(length(lines))
  for (i in seq_along(lines)) {
    is_hdr <- .is_section_header(lines[[i]])
    if (is_hdr) {
      in_section <- grepl(section_match, lines[[i]], fixed = TRUE)
      next  # drop the header itself
    }
    keep[[i]] <- in_section
  }
  lines[keep]
}

# Parse a single line into ms + throughput, honoring value_label when
# given so the right column is picked out of multi-number lines.
.parse_line <- function(line, value_label = NULL) {
  out <- list()
  m_ms <- regmatches(line,
                     regexec("([0-9.]+)\\s*ms", line, perl = TRUE))[[1]]
  if (length(m_ms) >= 2) out$ms <- as.numeric(m_ms[2])

  if (!is.null(value_label) && nzchar(value_label)) {
    # Number followed by literal value_label (e.g. 'dense-equiv GFLOPS').
    pat <- sprintf("([0-9][0-9,.]*)\\s*%s",
                   gsub("([][}{()|.+*?^$\\\\])", "\\\\\\1", value_label, perl = TRUE))
    m <- regmatches(line, regexec(pat, line, perl = TRUE,
                                  ignore.case = TRUE))[[1]]
    if (length(m) >= 2) {
      out$throughput <- as.numeric(gsub(",", "", m[2], fixed = TRUE))
      # Unit: take the trailing GFLOPS/TOPS word from value_label if
      # present, else default to GFLOPS.
      u <- regmatches(value_label,
                      regexec("(GFLOPS|TOPS)\\b", value_label,
                              perl = TRUE, ignore.case = TRUE))[[1]]
      out$unit <- if (length(u) >= 2) toupper(u[2]) else "GFLOPS"
    }
  } else {
    m_tp <- regmatches(line,
                       regexec("([0-9][0-9,.]*)\\s*(GFLOPS|TOPS)",
                               line, perl = TRUE, ignore.case = TRUE))[[1]]
    if (length(m_tp) >= 3) {
      out$throughput <- as.numeric(gsub(",", "", m_tp[2], fixed = TRUE))
      out$unit <- toupper(m_tp[3])
    }
  }
  out
}

# Pick the right line out of bench output, given a baseline-config's
# match/section directives. Returns the chosen line (string) or NULL.
.pick_line <- function(output_lines, match_str = NULL, section_str = NULL) {
  candidates <- .filter_section(output_lines, section_str)
  has_metrics <- grepl("ms", candidates) &
                 grepl("GFLOPS|TOPS", candidates, ignore.case = TRUE)
  candidates <- candidates[has_metrics]
  if (!length(candidates)) return(NULL)
  if (is.null(match_str) || !nzchar(match_str)) {
    return(candidates[[1]])  # legacy: first metric-bearing line
  }
  hits <- candidates[grepl(match_str, candidates, fixed = TRUE)]
  if (length(hits)) hits[[1]] else NULL
}

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
  out <- tryCatch(
    suppressWarnings(system2(abs_exe, args, stdout = TRUE, stderr = TRUE,
                             timeout = 120)),
    error = function(e) {
      attr(character(0), "status") <- 1L
      character(0)
    }
  )
  status <- attr(out, "status")
  rc <- if (is.null(status)) 0L else as.integer(status)
  output <- paste(out, collapse = "\n")

  metrics <- list(raw_output = output, returncode = rc)

  match_str   <- if (!is.null(baseline_cfg)) baseline_cfg$match       else NULL
  section_str <- if (!is.null(baseline_cfg)) baseline_cfg$section     else NULL
  value_label <- if (!is.null(baseline_cfg)) baseline_cfg$value_label else NULL

  picked <- .pick_line(out, match_str = match_str, section_str = section_str)
  if (is.null(picked)) {
    # Nothing matched our hints; fall back to whole-output scan as before.
    picked <- output
  }
  parsed <- .parse_line(picked, value_label = value_label)
  metrics$ms         <- parsed$ms
  metrics$throughput <- parsed$throughput
  metrics$unit       <- parsed$unit
  metrics$matched_line <- picked
  metrics
}

# ----------------------------------------------------------------------
# Regression decision
# ----------------------------------------------------------------------
check_regression <- function(current, baseline, tolerance) {
  if (!is.null(current$returncode) && current$returncode != 0L) {
    return(list(is_reg = TRUE,
                msg = sprintf("CRASH (exit=%d)", current$returncode)))
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
    cat("Run benchmarks manually and record results to docs/baselines.json\n")
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

  regressions <- 0L; improvements <- 0L; total <- 0L

  # Reserved keys at the kernel-entry level that are not config names.
  RESERVED_KEYS <- c("exe")

  for (kernel_path in names(kernels)) {
    entry <- kernels[[kernel_path]]
    # `exe` override (Tier 9 schema): use it if present, else heuristic.
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
      current <- run_benchmark(exe, cfg_args, baseline_cfg = baseline_cfg)
      verdict <- check_regression(current, baseline_cfg, args$tolerance)

      cat(sprintf("\n%s [%s]\n  %s\n", kernel_path, cfg, verdict$msg))
      if (verdict$is_reg) regressions <- regressions + 1L
      else if (grepl("IMPROVED", verdict$msg, fixed = TRUE)) improvements <- improvements + 1L
    }
  }

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("  Total: %d | Regressions: %d | Improvements: %d\n",
              total, regressions, improvements))
  if (regressions > 0L) {
    cat(sprintf("  RESULT: FAILED -- %d regression(s) detected\n", regressions))
    quit(status = 1)
  } else {
    cat("  RESULT: PASSED -- all benchmarks within tolerance\n")
    quit(status = 0)
  }
}

if (sys.nframe() == 0L) main()

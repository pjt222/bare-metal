#!/usr/bin/env Rscript
# scripts/bench/bench_all_collect.R
#
# Materialise bench_all's JSONL sample store(s) into a single RDS
# data.table rollup for analysis / plotting (issue #152, Q3).
#
# JSONL is the source of truth -- append-only, one row per attempt,
# atomic appends, Ctrl+C-safe. The RDS is a DERIVED artifact, regenerable
# from JSONL at any time; never edit it, never treat it as authoritative.
#
# The counterpart for the grid sweep is scripts/probe/grid_collect.R.
# This script is the converged replacement: same JSONL-primary contract,
# same tolerant read, but it rolls up EVERY run directory under
# results/bench_all/ into one cross-run table keyed
# (git_head, cell_id, regime).
#
# Schema drift is expected and handled: runs recorded before a column
# existed simply lack it, and `rbindlist(fill = TRUE)` fills NA. That is
# why the rollup must never be built with rbind() or do.call(rbind, ...).
#
# Usage:
#   Rscript scripts/bench/bench_all_collect.R [--print]
#   Rscript scripts/bench/bench_all_collect.R --jsonl PATH [--jsonl PATH2]
#   Rscript scripts/bench/bench_all_collect.R --results-dir DIR --out PATH
#
# Defaults:
#   --results-dir results/bench_all
#   --out         results/bench_all/bench_all_results.rds
#
# --print emits a stdout summary (rows, valid count, cells, regimes,
# per-(cell, regime) median throughput). GPU-free -- safe in CI and in
# the pre-push gate.

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(cuasmR)   # tolerant JSONL reader (issue #134)
})

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

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
    if (parent == cur) { cur <- start; break }
    cur <- parent
  }
  cur
}

parse_args <- function(argv) {
  out <- list(results_dir = file.path(REPO_ROOT, "results", "bench_all"),
              jsonl = character(0), out = NULL, print = FALSE)
  need <- function(a, i) {
    if (i + 1L > length(argv)) stop(sprintf("%s needs a value", a))
    argv[i + 1L]
  }
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if      (a == "--results-dir") { out$results_dir <- need(a, i); i <- i+2 }
    else if (a == "--jsonl")       { out$jsonl <- c(out$jsonl, need(a, i)); i <- i+2 }
    else if (a == "--out")         { out$out <- need(a, i); i <- i+2 }
    else if (a == "--print")       { out$print <- TRUE; i <- i+1 }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: bench_all_collect.R [--results-dir D] [--jsonl F]... ",
          "[--out F] [--print]\n",
          "  Rolls every results/bench_all/*/samples.jsonl into one RDS.\n",
          "  JSONL is the source of truth; the RDS is regenerable.\n", sep = "")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  # An explicit --jsonl is a targeted collect over hand-picked files. It
  # must NOT silently overwrite the canonical cross-run rollup with a
  # partial (or foreign-store) view -- make the caller name the output.
  if (length(out$jsonl) && is.null(out$out))
    stop("--jsonl requires an explicit --out (refusing to overwrite the canonical rollup)")
  out$out <- out$out %||% file.path(out$results_dir, "bench_all_results.rds")
  out
}

# ----------------------------------------------------------------------
# Pure functions (GPU-free, unit-tested in tests/bench_all/).
# ----------------------------------------------------------------------

#' Every samples.jsonl under a bench_all results directory, sorted so the
#' rollup is deterministic regardless of filesystem order.
discover_jsonl <- function(results_dir) {
  if (!dir.exists(results_dir)) return(character(0))
  sort(list.files(results_dir, pattern = "^samples\\.jsonl$",
                  recursive = TRUE, full.names = TRUE))
}

#' Bind a list of per-row lists into a data.table.
#'
#' `fill = TRUE` is load-bearing: it is what lets a rollup span runs
#' recorded before/after a schema addition (e.g. the #152 regime columns)
#' without dropping either side. Returns an empty data.table for no rows.
bind_rows_fill <- function(rows) {
  if (!length(rows)) return(data.table())
  rbindlist(lapply(rows, function(x) as.data.table(x)), fill = TRUE)
}

#' Ensure the #152 regime columns exist and are typed, so a rollup that
#' contains only pre-#152 runs still groups correctly.
#'
#' A bench_all row recorded before regimes existed was, by construction,
#' a native run -- bench_all had no other mode -- so backfilling `regime`
#' to "native" is a statement of fact there.
#'
#' It is NOT a fact for rows from another store. `--jsonl` accepts any
#' path, and grid rows carry `clock_target_mhz` with no `regime` key, so
#' an unconditional backfill would relabel 1200/1410/1605 MHz samples as
#' native and let the summary average them into one bucket. Derive the
#' label from `clock_target_mhz` whenever the row has one; only a row
#' with no clock target at all is assumed native.
normalise_schema <- function(dt) {
  if (nrow(dt) == 0L) return(dt)
  if (!"regime" %in% names(dt)) dt[, regime := NA_character_]
  if ("clock_target_mhz" %in% names(dt))
    dt[is.na(regime) & !is.na(clock_target_mhz),
       regime := as.character(as.integer(clock_target_mhz))]
  dt[is.na(regime), regime := "native"]
  for (col in c("clock_target_mhz", "band_lo", "band_hi", "clock_mem_mhz"))
    if (!col %in% names(dt)) dt[, (col) := NA_integer_]
  for (col in c("measurable", "verified"))
    if (!col %in% names(dt)) dt[, (col) := NA]
  dt[]
}

load_samples_dt <- function(paths) {
  rows <- list()
  n_total <- 0L; n_bad <- 0L
  for (p in paths) {
    if (!file.exists(p)) { cat(sprintf("WARN: missing JSONL: %s\n", p)); next }
    r <- cuasmR::read_jsonl(p, simplify = TRUE)
    n_total <- n_total + r$n_total
    n_bad   <- n_bad + r$n_bad
    if (r$n_bad > 0L)
      cat(sprintf("WARN: %d/%d line(s) failed to parse (truncated tail?): %s\n",
                  r$n_bad, r$n_total, p))
    rows <- c(rows, r$rows)
  }
  cat(sprintf("Read %d row(s) from %d file(s)%s.\n", n_total, length(paths),
              if (n_bad) sprintf(" (%d unparseable)", n_bad) else ""))
  normalise_schema(bind_rows_fill(rows))
}

summarise <- function(dt) {
  if (nrow(dt) == 0L) return(invisible())
  cat("\n", strrep("=", 72), "\n", sep = "")
  cat("  bench_all_collect summary\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat(sprintf("  Total rows        : %d\n", nrow(dt)))
  cat(sprintf("  Valid samples     : %d / %d\n", sum(dt$valid %in% TRUE), nrow(dt)))
  cat(sprintf("  Distinct cells    : %d\n", length(unique(dt$cell_id))))
  cat(sprintf("  Distinct regimes  : %s\n", paste(sort(unique(dt$regime)), collapse = ", ")))
  cat(sprintf("  Distinct git HEADs: %d\n", length(unique(dt$git_head))))
  cat(sprintf("  Run IDs           : %d\n", length(unique(dt$run_id))))

  ok <- dt[valid %in% TRUE]
  if (nrow(ok) == 0L) { cat("\n  No valid samples.\n"); return(invisible()) }

  # A valid row with NO throughput is the once-native confirming run of a
  # `measurable: false` bench -- it ran fine and has nothing to average.
  # Summarising it as a number would print Inf/-Inf spreads and imply a
  # measurement that was never claimed, so count it separately.
  numeric_ok <- ok[!is.na(throughput)]
  n_nonnum   <- nrow(ok) - nrow(numeric_ok)

  # Grouped by the DECLARED store key (git_head, cell_id, regime), not by
  # (cell_id, regime). Dropping git_head would blend a pre- and a
  # post-optimization commit into one median -- the single most
  # misleading number this script could print.
  n_heads <- length(unique(numeric_ok$git_head))
  cat(sprintf("\n  Per-(git_head, cell, regime) medians -- valid samples with a number\n"))
  summ <- numeric_ok[, .(n = .N,
                         tput_med = stats::median(throughput, na.rm = TRUE),
                         tput_min = min(throughput, na.rm = TRUE),
                         tput_max = max(throughput, na.rm = TRUE),
                         ms_med   = stats::median(ms, na.rm = TRUE),
                         clk_med  = suppressWarnings(
                           as.integer(stats::median(clock_sm_mhz, na.rm = TRUE)))),
                     by = .(git_head, cell_id, regime, unit)][order(cell_id, regime, git_head)]
  show_head <- n_heads > 1L
  for (i in seq_len(nrow(summ))) {
    r <- summ[i]
    cat(sprintf("    %-30s @ %-8s %sn=%-3d median %10s %-18s spread %s-%s  obs_clk %s\n",
                r$cell_id, r$regime,
                if (show_head) sprintf("[%s] ", substr(r$git_head, 1, 7)) else "",
                r$n,
                format(round(r$tput_med, 0), big.mark = ""),
                if (is.na(r$unit)) "" else r$unit,
                format(round(r$tput_min, 0), big.mark = ""),
                format(round(r$tput_max, 0), big.mark = ""),
                if (is.na(r$clk_med)) "?" else as.character(r$clk_med)))
  }
  if (show_head)
    cat(sprintf("\n  %d git HEADs present -- rows are NOT comparable across them.\n", n_heads))
  if (n_nonnum > 0L)
    cat(sprintf("\n  %d valid row(s) carry no throughput (non-measurable confirming runs).\n",
                n_nonnum))

  bad <- dt[!(valid %in% TRUE)]
  if (nrow(bad) > 0L) {
    cat("\n  Reject reasons\n")
    rj <- bad[, .N, by = reject_reason][order(-N)]
    for (i in seq_len(nrow(rj)))
      cat(sprintf("    %5d  %s\n", rj$N[i], rj$reject_reason[i]))
  }
  cat(strrep("=", 72), "\n", sep = "")
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  paths <- if (length(args$jsonl)) args$jsonl else discover_jsonl(args$results_dir)
  if (!length(paths)) {
    cat(sprintf("No samples.jsonl found under %s. Nothing to collect.\n",
                args$results_dir))
    quit(status = 0)
  }
  dt <- load_samples_dt(paths)
  if (nrow(dt) == 0L) {
    cat("No rows materialised. Not writing RDS.\n")
    quit(status = 0)
  }
  dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)
  saveRDS(dt, args$out)
  cat(sprintf("Materialised %d row(s) -> %s\n", nrow(dt), args$out))
  if (args$print) summarise(dt)
  invisible(NULL)
}

if (sys.nframe() == 0L) main()

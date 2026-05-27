#!/usr/bin/env Rscript
# scripts/probe/grid_collect.R
#
# Materialise the JSONL primary store (one row per sample, atomic
# appends) into an RDS data.table view for analysis / plotting.
#
# JSONL is the source of truth — append-only and Ctrl+C-safe. RDS is a
# derived artifact, regenerable from JSONL at any time.
#
# Usage:
#   Rscript scripts/probe/grid_collect.R \
#     [--jsonl PATH] [--out PATH] [--print]
#
# Defaults:
#   --jsonl scripts/probe/eval_logs/grid_sweep_samples.jsonl
#   --out   scripts/probe/eval_logs/grid_sweep_results.rds
#
# --print emits a short stdout summary (rows, valid count, cells,
# clock groups, per-cell median throughput). Useful for CI smoke
# checks and quick post-sweep verification without loading R
# interactively.

suppressPackageStartupMessages({
  library(here)
  library(fs)
  library(cli)
  library(data.table)
  library(jsonlite)
})

parse_args <- function(argv) {
  out <- list(
    jsonl = here("scripts", "probe", "eval_logs",
                 "grid_sweep_samples.jsonl"),
    out   = here("scripts", "probe", "eval_logs",
                 "grid_sweep_results.rds"),
    print = FALSE
  )
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    switch(a,
      "--jsonl" = { out$jsonl <- argv[i + 1]; i <- i + 2 },
      "--out"   = { out$out   <- argv[i + 1]; i <- i + 2 },
      "--print" = { out$print <- TRUE; i <- i + 1 },
      stop(sprintf("unknown arg: %s", a))
    )
  }
  out
}

read_jsonl <- function(path) {
  if (!file_exists(path)) {
    cli_abort("JSONL not found: {.path {path}}")
  }
  lines <- readLines(path, warn = FALSE)
  if (length(lines) == 0L) {
    cli_alert_warning("JSONL is empty: {.path {path}}")
    return(data.table())
  }

  # Tolerant parser: a partially-written final line from a hard kill
  # is the documented failure mode. Drop unparseable rows; report
  # how many were dropped.
  parsed <- lapply(lines, function(l) {
    tryCatch(fromJSON(l, simplifyVector = TRUE),
             error = function(e) NULL)
  })
  ok <- !vapply(parsed, is.null, logical(1))
  n_bad <- sum(!ok)
  if (n_bad > 0L) {
    cli_alert_warning(
      "{n_bad}/{length(lines)} JSONL line(s) failed to parse (truncated tail?)")
  }
  rows <- parsed[ok]

  # Each fromJSON list -> single-row data.table. rbindlist with
  # fill=TRUE handles schema additions over time.
  dt <- rbindlist(lapply(rows, function(r) as.data.table(r)),
                  fill = TRUE)
  dt
}

summarise <- function(dt) {
  if (nrow(dt) == 0L) return(invisible())
  cli_h1("grid_collect summary")
  cli_text("Total rows           : {nrow(dt)}")
  cli_text("Valid samples        : {sum(dt$valid)} / {nrow(dt)}")
  cli_text("Distinct cells       : {length(unique(dt$cell_id))}")
  cli_text("Distinct clock targets: {length(unique(dt$clock_target_mhz))}")
  cli_text("Distinct git HEADs   : {length(unique(dt$git_head))}")
  cli_text("Run IDs              : {length(unique(dt$run_id))}")

  cli_h2("Per-(cell, clock) summary (valid samples only)")
  ok <- dt[valid == TRUE]
  if (nrow(ok) == 0L) {
    cli_alert_warning("No valid samples in JSONL.")
    return(invisible())
  }
  summ <- ok[, .(
    n          = .N,
    tput_med   = stats::median(throughput, na.rm = TRUE),
    tput_min   = min(throughput, na.rm = TRUE),
    tput_max   = max(throughput, na.rm = TRUE),
    ms_med     = stats::median(ms, na.rm = TRUE),
    clk_med    = as.integer(stats::median(clock_observed_mhz, na.rm = TRUE))
  ), by = .(cell_id, clock_target_mhz, unit)][order(cell_id, clock_target_mhz)]

  for (i in seq_len(nrow(summ))) {
    r <- summ[i]
    clk_lbl <- if (is.na(r$clock_target_mhz)) "native"
               else sprintf("%d MHz", r$clock_target_mhz)
    cli_text(sprintf(
      "  %-30s @ %-9s : n=%d  median %s %s  spread %s-%s  obs_clk %d",
      r$cell_id, clk_lbl, r$n,
      format(round(r$tput_med, 0), big.mark = ""),
      r$unit,
      format(round(r$tput_min, 0), big.mark = ""),
      format(round(r$tput_max, 0), big.mark = ""),
      r$clk_med
    ))
  }

  if (nrow(dt[valid == FALSE]) > 0L) {
    cli_h2("Reject reasons")
    rj <- dt[valid == FALSE, .N, by = reject_reason][order(-N)]
    for (i in seq_len(nrow(rj))) {
      cli_text(sprintf("  %5d  %s", rj$N[i], rj$reject_reason[i]))
    }
  }
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dt <- read_jsonl(args$jsonl)
  if (nrow(dt) == 0L) {
    cli_alert_warning("No rows materialised. Not writing RDS.")
    return(invisible())
  }
  saveRDS(dt, args$out)
  cli_alert_success(
    "Materialised {nrow(dt)} rows -> {.path {args$out}}")
  if (args$print) summarise(dt)
}

if (sys.nframe() == 0L) main()

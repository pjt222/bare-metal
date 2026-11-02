#!/usr/bin/env Rscript
# scripts/probe/conv2d_variance.R
#
# Characterization harness for #195: why does conv2d_implicit_gemm span
# 1.36x run-to-run with no distinguishing condition in the record?
#
# This is a PROBE, not a gate. It measures nothing for pass/fail; it
# records every attempt with the conditions around it so the
# distribution can be read afterwards.
#
# Why not grid_measure.R (#135)? Three gaps, and they are precisely the
# instruments this question needs:
#   - grid records ONE parsed number per run. Here we parse SIX (explicit
#     and implicit for all three SD configs) from the same process. The
#     bench measures them in a fixed order inside one process, so the
#     within-run pattern locates a loss on the timeline for free: if the
#     implicit number swings while the explicit number measured ~200ms
#     earlier stays tight, the cause is local to that kernel's window and
#     not to the machine's global state.
#   - grid has no inter-sample gap. The gate's real runs are minutes to
#     hours apart; back-to-back sampling measures the GPU in its steadiest
#     possible state and may never sample the tail that blocks pushes.
#   - grid records the post snapshot only. A pre/post pair is what
#     bench_regress actually uses, so that is what has to be judged.
# grid's schema and resume key are locked at #135 and #160/#165 already
# track divergence, so it is not the thing to bend mid-characterization.
#
# Conditions, interleaved in blocks so thermal and time drift hit both:
#   A  back-to-back, no gap   (warm/steady)
#   B  gate-like idle gap     (--gap seconds before each sample)
#
# Usage:
#   Rscript scripts/probe/conv2d_variance.R --n 15 --gap 90 --block 1 \
#     --jsonl results/probe/conv2d_variance.jsonl
#
# --block must be < --n, or the run is refused (#212): with block >= n the
# sequence degenerates to AAA...BBB, balanced but not interleaved.
# --burnin K (default 0) runs K warm-up samples, condition W with no gap,
# before the A/B sequence. They are recorded with burnin = TRUE so a
# consumer can exclude them. Choose K before the run.
# An existing --jsonl is refused unless --append is given: a new campaign
# goes to a new file, so two runs are never pooled by accident. --append
# pools deliberately; it does not resume.
#
# Output: one JSONL row per sample per parsed measurement.

suppressPackageStartupMessages({
  library(here)
  library(fs)
  library(jsonlite)
})

# WSL CUDA libs FIRST on LD_LIBRARY_PATH. #189: this is a POSITION
# requirement, not a presence one -- with the native
# /usr/lib/x86_64-linux-gnu ahead of it, libnvidia-ml.so shadows the WSL
# copy and nvidia-smi exits 9, so every capture returns NULL.
local({
  wsl_lib <- "/usr/lib/wsl/lib"
  cur <- Sys.getenv("LD_LIBRARY_PATH")
  if (dir.exists(wsl_lib) && !startsWith(cur, wsl_lib)) {
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                   paste(wsl_lib, cur, sep = ":")
                                 else wsl_lib)
  }
})

suppressMessages(library(cuasmR))

`%||%` <- function(a, b) if (is.null(a)) b else a

parse_args <- function(argv) {
  # block = 1 (ABAB...) by default, not 5: a transient at the very start
  # of a session lands on whatever comes first, and with block 5 the first
  # five samples are all condition A (#212). Block interleaving spreads
  # slow drift; only alternation spreads a lead-in transient.
  out <- list(n = 15L, gap = 90, block = 1L, burnin = 0L, exe = NULL, append = FALSE,
              jsonl = "results/probe/conv2d_variance.jsonl")
  i <- 1
  while (i <= length(argv)) {
    switch(argv[i],
      "--n"     = { out$n     <- as.integer(argv[i + 1]); i <- i + 2 },
      "--gap"   = { out$gap   <- as.numeric(argv[i + 1]); i <- i + 2 },
      "--block" = { out$block <- as.integer(argv[i + 1]); i <- i + 2 },
      "--burnin" = { out$burnin <- as.integer(argv[i + 1]); i <- i + 2 },
      "--jsonl" = { out$jsonl <- argv[i + 1];             i <- i + 2 },
      "--exe"   = { out$exe   <- argv[i + 1];             i <- i + 2 },
      "--append" = { out$append <- TRUE;                  i <- i + 1 },
      stop(sprintf("unknown arg: %s", argv[i])))
  }
  out
}

# The six numbers the bench prints, in the order it measures them. The
# ORDER is the point: a loss that appears only in later entries is a
# different animal from one that hits the whole process.
MEASUREMENTS <- list(
  list(key = "sd64_explicit",  section = "SD 64",     match = "Explicit (im2col+GEMM)", ord = 1L),
  list(key = "sd64_implicit",  section = "SD 64",     match = "Implicit (single kern)", ord = 2L),
  list(key = "sd32_explicit",  section = "SD 32",     match = "Explicit (im2col+GEMM)", ord = 3L),
  list(key = "sd32_implicit",  section = "SD 32",     match = "Implicit (single kern)", ord = 4L),
  list(key = "sd128_explicit", section = "SD 128",    match = "Explicit (im2col+GEMM)", ord = 5L),
  list(key = "sd128_implicit", section = "SD 128",    match = "Implicit (single kern)", ord = 6L)
)

# The snapshot fields recorded per sample, by the names capture_gpu_state()
# emits (R/cuasmR/R/bench_meta.R). Both power references are kept, for the
# reason #209 keeps both: "below default" alone misses the 115 W fallback
# on this laptop, and "below max" alone false-positives on hardware whose
# normal point is the default.
SNAPSHOT_GPU_FIELDS <- c("clock_sm", "clock_mem", "temp_c", "power_w",
                         "pstate", "throttle", "power_limit_w",
                         "power_limit_default_w", "power_limit_max_w",
                         "power_below_default", "power_below_max")
SNAPSHOT_LOAD_FIELDS <- c("load_1m", "load_5m", "load_15m")

# Strict on NAMES, lenient on VALUES (#213). This used to read
# `g$power_capped %||% NA`, a field capture_gpu_state() stopped emitting
# at #207/#209, so the column was NA on 100% of the pilot's rows and
# nothing complained. capture_gpu_state() always creates every name, and
# a value the driver did not report arrives as NULL or NA, which is
# recorded as NA. A NAME that is absent means this script and cuasmR
# disagree on the schema, and the run stops before writing rows that only
# look complete.
#
# `snapshot` separates "no snapshot at all" (capture_gpu_state() returned
# NULL, e.g. nvidia-smi failed) from "a snapshot whose values are NA".
# Without it the two are indistinguishable in the JSONL.
#
# Host loadavg is recorded too (#213 defect 2). The pilot's contamination
# (load 2.99 -> 12.76) was invisible per sample, so it could not be
# regressed out.
gpu_fields <- function(snap, prefix) {
  out_names <- c("snapshot", SNAPSHOT_GPU_FIELDS, SNAPSHOT_LOAD_FIELDS)
  out <- setNames(rep(list(NA), length(out_names)), out_names)
  out$snapshot <- !is.null(snap)
  if (!is.null(snap)) {
    g <- snap$gpu
    missing_gpu <- setdiff(SNAPSHOT_GPU_FIELDS, names(g))
    if (length(missing_gpu)) {
      stop(sprintf(paste0("capture_gpu_state() snapshot has no field(s) %s -- ",
                          "this script and cuasmR disagree on the schema (#213); ",
                          "fix the names here rather than recording NA"),
                   paste(missing_gpu, collapse = ", ")), call. = FALSE)
    }
    for (f in setdiff(SNAPSHOT_GPU_FIELDS, "throttle")) {
      out[[f]] <- g[[f]] %||% NA
    }
    # character(0) = no active reason -> ""; NA = an unreadable mask (#211)
    # -> NA, not the literal string "NA" that paste() would produce.
    thr <- g$throttle
    out$throttle <- if (length(thr) == 0L) "" else if (anyNA(thr)) NA
                    else paste(thr, collapse = ",")
    la <- snap$host$loadavg
    if (!is.null(la)) {
      missing_load <- setdiff(SNAPSHOT_LOAD_FIELDS, names(la))
      if (length(missing_load)) {
        stop(sprintf("snapshot loadavg has no field(s) %s (#213)",
                     paste(missing_load, collapse = ", ")), call. = FALSE)
      }
      for (f in SNAPSHOT_LOAD_FIELDS) out[[f]] <- la[[f]] %||% NA
    }
  }
  setNames(out, paste0(prefix, out_names))
}

# Interleave A/B in blocks so thermal drift and any slow platform state
# hit both conditions rather than confounding one of them.
#
# Draw from a per-condition remaining count rather than building a
# pattern and truncating it. Truncation silently unbalances the design
# whenever n is not a multiple of block -- n=15, block=4 gives 16 A and
# 14 B, and an unbalanced A/B is exactly the defect this harness exists
# to avoid. Guaranteed here: exactly `n` of each, for any block size.
#
# Balance is necessary, not sufficient (#212). With block >= n the loop
# takes min(block, n) = n per condition and emits AAA...BBB: perfectly
# balanced, so a balance check passes, but there is no interleave at all
# and condition is confounded with time order. That design is rejected
# here rather than run, and the result must switch condition more than
# once.
assign_conditions <- function(n, block) {
  if (length(n) != 1L || is.na(n) || n < 2L) {
    stop(sprintf("--n must be an integer >= 2 (got %s): fewer than two samples per condition cannot interleave",
                 format(n)), call. = FALSE)
  }
  if (length(block) != 1L || is.na(block) || block < 1L) {
    stop(sprintf("--block must be an integer >= 1 (got %s)", format(block)),
         call. = FALSE)
  }
  if (block >= n) {
    stop(sprintf(paste0("--block %d >= --n %d: each condition would run as one ",
                        "contiguous block (%s%s), which confounds condition with ",
                        "time order. Use --block < --n; --block 1 gives ABAB..."),
                 block, n, strrep("A", n), strrep("B", n)), call. = FALSE)
  }
  left <- c(A = n, B = n)
  conds <- character(0)
  while (sum(left) > 0L) {
    for (cd in c("A", "B")) {
      take <- min(block, left[[cd]])
      if (take > 0L) {
        conds <- c(conds, rep(cd, take))
        left[[cd]] <- left[[cd]] - take
      }
    }
  }
  stopifnot(sum(conds == "A") == n, sum(conds == "B") == n,
            length(rle(conds)$lengths) > 2L)
  conds
}

# Refuse to write into an existing JSONL unless --append says so (#224
# review, H1). append_jsonl_row() appends, and the default --jsonl is the
# committed evidence file of the 2026-11-02 campaign. A re-run would
# silently pool two campaigns into one file. A new campaign gets a new
# file. --append means DELIBERATE pooling: it does not resume a run (a new
# run_id is minted and sample_idx restarts at 1), and the analysis script
# refuses a file holding more than one run_id.
check_output_path <- function(path, append) {
  if (file.exists(path) && !isTRUE(append)) {
    stop(sprintf(paste0("%s already exists. Write a new campaign to a new --jsonl, ",
                        "or pass --append to add to this one deliberately."), path),
         call. = FALSE)
  }
  invisible(path)
}

# Lead-in handling, explicit and recorded (#212 defect 2). Interleaving
# spreads slow drift but not a transient at the very start of a session,
# because nothing precedes it. `--burnin K` runs K warm-up samples
# (condition "W", no gap) BEFORE the A/B sequence. They are written to the
# JSONL with burnin = TRUE, never silently dropped, so a consumer can
# exclude them and can also measure whether the lead-in transient exists
# at all. K is a pre-registered choice: set it before the run, never after
# seeing the data. The default 0 leaves the design unchanged, and the
# rows still say so (burnin = FALSE, burnin_n = 0).
with_burnin <- function(conds, burnin) {
  if (length(burnin) != 1L || is.na(burnin) || burnin < 0L) {
    stop(sprintf("--burnin must be an integer >= 0 (got %s)", format(burnin)),
         call. = FALSE)
  }
  c(rep("W", burnin), conds)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  exe <- args$exe %||% here("kernels/convolution/conv2d/bench_implicit_gemm")
  exe <- path_abs(exe)
  if (!file_exists(exe)) stop(sprintf("exe not found: %s", exe))

  check_output_path(args$jsonl, args$append)
  dir_create(path_dir(args$jsonl), recurse = TRUE)

  run_id <- format(Sys.time(), "%Y%m%dT%H%M%S")
  gh <- tryCatch(trimws(system2("git", c("rev-parse", "--short", "HEAD"),
                                stdout = TRUE, stderr = FALSE)[1]),
                 error = function(e) NA_character_)
  policy <- tryCatch(capture_power_policy(), error = function(e) NULL)

  conds <- with_burnin(assign_conditions(args$n, args$block), args$burnin)

  message(sprintf("probe #195: %d samples (%d burn-in + %d per condition), gap=%.0fs, block=%d",
                  length(conds), args$burnin, args$n, args$gap, args$block))
  message(sprintf("  exe    : %s", exe))
  message(sprintf("  jsonl  : %s", args$jsonl))
  message(sprintf("  overlay: %s", if (is.null(policy)) "unknown"
                                   else policy$overlay_name))

  prev_wd <- getwd()
  on.exit(setwd(prev_wd), add = TRUE)

  for (i in seq_along(conds)) {
    cond <- conds[[i]]

    # Condition B reproduces the gate's real cadence: the GPU is allowed
    # to fall back to idle before the sample, the way it does between two
    # pushes minutes apart.
    if (identical(cond, "B") && args$gap > 0) {
      message(sprintf("[%d/%d] cond B: idling %.0fs", i, length(conds), args$gap))
      Sys.sleep(args$gap)
    }

    setwd(path_dir(exe))
    r <- run_bench(exe, character(0), timeout = 300)
    setwd(prev_wd)

    if (identical(r$rc, 130L)) {
      message("bench exited 130 (SIGINT) -- user cancel")
      quit(save = "no", status = 130L)
    }

    ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC")

    for (m in MEASUREMENTS) {
      p <- parse_throughput(r$out, match = m$match, section = m$section,
                            pick = "first")
      row <- c(
        list(run_id = run_id, ts_utc = ts, git_head = gh,
             sample_idx = i, condition = cond,
             burnin = identical(cond, "W"), burnin_n = args$burnin,
             block = args$block, gap_s = args$gap,
             measurement = m$key, order_in_run = m$ord,
             ms = p$ms, throughput = p$throughput, unit = p$unit,
             rc = r$rc,
             overlay = if (is.null(policy)) NA_character_
                       else policy$overlay_name),
        gpu_fields(r$pre,  "pre_"),
        gpu_fields(r$post, "post_"))
      append_jsonl_row(args$jsonl, row)
    }

    imp <- parse_throughput(r$out, match = "Implicit (single kern)",
                            section = "SD 64", pick = "first")
    message(sprintf("[%d/%d] cond %s  sd64_implicit = %s GFLOPS  (post clk=%s power=%sW)",
                    i, length(conds), cond,
                    if (is.na(imp$throughput)) "NA"
                    else format(round(imp$throughput)),
                    r$post$gpu$clock_sm %||% NA,
                    r$post$gpu$power_w %||% NA))
  }

  message(sprintf("done: %d samples -> %s", length(conds), args$jsonl))
}

if (sys.nframe() == 0L) {
  tryCatch(main(), interrupt = function(c) {
    message("Interrupted by user (SIGINT)")
    quit(save = "no", status = 130L)
  })
}

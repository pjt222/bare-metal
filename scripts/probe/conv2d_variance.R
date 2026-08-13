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
#   Rscript scripts/probe/conv2d_variance.R --n 15 --gap 90 --block 5 \
#     --jsonl results/probe/conv2d_variance.jsonl
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
  out <- list(n = 15L, gap = 90, block = 5L, exe = NULL,
              jsonl = "results/probe/conv2d_variance.jsonl")
  i <- 1
  while (i <= length(argv)) {
    switch(argv[i],
      "--n"     = { out$n     <- as.integer(argv[i + 1]); i <- i + 2 },
      "--gap"   = { out$gap   <- as.numeric(argv[i + 1]); i <- i + 2 },
      "--block" = { out$block <- as.integer(argv[i + 1]); i <- i + 2 },
      "--jsonl" = { out$jsonl <- argv[i + 1];             i <- i + 2 },
      "--exe"   = { out$exe   <- argv[i + 1];             i <- i + 2 },
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

gpu_fields <- function(snap, prefix) {
  if (is.null(snap)) {
    return(setNames(as.list(rep(NA, 8)),
                    paste0(prefix, c("clock_sm", "clock_mem", "temp_c",
                                     "power_w", "pstate", "throttle",
                                     "power_limit_w", "power_capped"))))
  }
  g <- snap$gpu
  out <- list(g$clock_sm, g$clock_mem, g$temp_c, g$power_w, g$pstate,
              paste(g$throttle, collapse = ","),
              g$power_limit_w %||% NA, g$power_capped %||% NA)
  names(out) <- paste0(prefix, c("clock_sm", "clock_mem", "temp_c",
                                 "power_w", "pstate", "throttle",
                                 "power_limit_w", "power_capped"))
  out
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  exe <- args$exe %||% here("kernels/convolution/conv2d/bench_implicit_gemm")
  exe <- path_abs(exe)
  if (!file_exists(exe)) stop(sprintf("exe not found: %s", exe))

  dir_create(path_dir(args$jsonl), recurse = TRUE)

  run_id <- format(Sys.time(), "%Y%m%dT%H%M%S")
  gh <- tryCatch(trimws(system2("git", c("rev-parse", "--short", "HEAD"),
                                stdout = TRUE, stderr = FALSE)[1]),
                 error = function(e) NA_character_)
  policy <- tryCatch(capture_power_policy(), error = function(e) NULL)

  # Interleave A/B in blocks so thermal drift and any slow platform state
  # hit both conditions rather than confounding one of them.
  #
  # Draw from a per-condition remaining count rather than building a
  # pattern and truncating it. Truncation silently unbalances the design
  # whenever n is not a multiple of block -- n=15, block=4 gives 16 A and
  # 14 B, and an unbalanced A/B is exactly the defect this harness exists
  # to avoid. Guaranteed here: exactly `n` of each, for any block size.
  left <- c(A = args$n, B = args$n)
  conds <- character(0)
  cond_order <- c("A", "B")
  while (sum(left) > 0L) {
    for (cd in cond_order) {
      take <- min(args$block, left[[cd]])
      if (take > 0L) {
        conds <- c(conds, rep(cd, take))
        left[[cd]] <- left[[cd]] - take
      }
    }
  }
  stopifnot(sum(conds == "A") == args$n, sum(conds == "B") == args$n)

  message(sprintf("probe #195: %d samples (%d per condition), gap=%.0fs, block=%d",
                  length(conds), args$n, args$gap, args$block))
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
             sample_idx = i, condition = cond, gap_s = args$gap,
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

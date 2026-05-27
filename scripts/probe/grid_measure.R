#!/usr/bin/env Rscript
# scripts/probe/grid_measure.R
#
# Multi-kernel × clock grid sweep — R brain (#135 Phase 2).
#
# Two subcommands:
#
#   plan     read grid_sweep.yml, expand into ordered cell list,
#            optionally filter cells already measured in a prior
#            JSONL store (resume support). Emits JSON to stdout —
#            the PowerShell orchestrator consumes this.
#
#   measure  measure one cell. Identified by --cell-id + --clock-target
#            (an int MHz or the literal string `native`). Appends one
#            JSONL row per sample to --jsonl. Caller is responsible
#            for applying / releasing the host-side clock lock; the
#            band-check fires only when --clock-target is an int and
#            the observed clock leaves the band, rejecting the sample.
#
# Persistence model: JSONL primary (atomic append, one line per
# sample, resilient to Ctrl+C mid-run). A separate grid_collect.R
# (Phase 4) materialises the rds derived view. Resume key is the
# three-tuple (git_head, clock_target_mhz, cell_id) — locked at this
# Phase 2 commit and documented in grid_sweep_methodology.md.

suppressPackageStartupMessages({
  library(here)
  library(fs)
  library(cli)
  library(data.table)
  library(yaml)
  library(jsonlite)
})

# WSL CUDA libs on LD_LIBRARY_PATH so a fresh Rscript can resolve
# nvidia-smi / the CUDA driver.
local({
  wsl_lib <- "/usr/lib/wsl/lib"
  cur <- Sys.getenv("LD_LIBRARY_PATH")
  if (dir_exists(wsl_lib) && !grepl(wsl_lib, cur, fixed = TRUE)) {
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                   paste(wsl_lib, cur, sep = ":")
                                 else wsl_lib)
  }
})

source(here("scripts", "bench", "bench_meta.R"))

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list(mode = NULL, spec = NULL, resume_jsonl = NULL,
              cell_id = NULL, clock_target = NULL, jsonl = NULL,
              run_id = NULL, out = NULL)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    switch(a,
      "--mode"         = { out$mode <- argv[i + 1]; i <- i + 2 },
      "--spec"         = { out$spec <- argv[i + 1]; i <- i + 2 },
      "--resume-jsonl" = { out$resume_jsonl <- argv[i + 1]; i <- i + 2 },
      "--cell-id"      = { out$cell_id <- argv[i + 1]; i <- i + 2 },
      "--clock-target" = { out$clock_target <- argv[i + 1]; i <- i + 2 },
      "--jsonl"        = { out$jsonl <- argv[i + 1]; i <- i + 2 },
      "--run-id"       = { out$run_id <- argv[i + 1]; i <- i + 2 },
      "--out"          = { out$out <- argv[i + 1]; i <- i + 2 },
      stop(sprintf("unknown arg: %s", a))
    )
  }
  out
}

# ----------------------------------------------------------------------
# Spec loading + validation
# ----------------------------------------------------------------------
load_spec <- function(path) {
  if (!file_exists(path)) stop(sprintf("spec not found: %s", path))
  spec <- read_yaml(path)

  for (req in c("defaults", "clocks", "kernels"))
    if (is.null(spec[[req]]))
      stop(sprintf("spec missing top-level field: %s", req))

  defaults <- spec$defaults
  for (req in c("n_samples", "warmup", "band_mhz"))
    if (is.null(defaults[[req]]))
      stop(sprintf("spec defaults missing: %s", req))

  # Normalise + validate kernels.
  ids <- character(length(spec$kernels))
  for (j in seq_along(spec$kernels)) {
    k <- spec$kernels[[j]]
    for (req in c("id", "exe", "match"))
      if (is.null(k[[req]]))
        stop(sprintf("kernel #%d missing field: %s", j, req))
    if (is.null(k$args)) k$args <- list()
    if (is.null(k$regimes)) k$regimes <- spec$clocks
    if (is.null(k$n_samples)) k$n_samples <- defaults$n_samples
    if (is.null(k$warmup))    k$warmup    <- defaults$warmup
    if (is.null(k$band_mhz))  k$band_mhz  <- defaults$band_mhz
    if (id_dupe <- (k$id %in% ids))
      stop(sprintf("duplicate kernel id: %s", k$id))
    ids[j] <- k$id
    spec$kernels[[j]] <- k
  }
  spec
}

normalise_clock <- function(c) {
  # `native` -> NA_integer_; int MHz -> integer; everything else stops.
  if (is.null(c)) return(NA_integer_)
  if (identical(c, "native")) return(NA_integer_)
  ci <- suppressWarnings(as.integer(c))
  if (is.na(ci) || ci < 100L || ci > 5000L)
    stop(sprintf("invalid clock value: %s", as.character(c)))
  ci
}

# ----------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------
load_jsonl_keys <- function(path) {
  if (is.null(path) || !file_exists(path)) return(character(0))
  lines <- readLines(path, warn = FALSE)
  if (length(lines) == 0L) return(character(0))
  rows <- lapply(lines, function(l) {
    tryCatch(fromJSON(l, simplifyVector = FALSE),
             error = function(e) NULL)
  })
  rows <- Filter(Negate(is.null), rows)
  keys <- vapply(rows, function(r) {
    clk <- if (is.null(r$clock_target_mhz) || is.na(r$clock_target_mhz))
             "native" else as.character(as.integer(r$clock_target_mhz))
    sprintf("%s|%s|%s", r$git_head %||% "", clk, r$cell_id %||% "")
  }, character(1))
  unique(keys)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

git_head <- function() {
  res <- tryCatch(system2("git", c("rev-parse", "HEAD"),
                          stdout = TRUE, stderr = FALSE),
                  error = function(e) NA_character_)
  if (length(res) == 0L) NA_character_ else res[[1]]
}

build_plan <- function(spec, resume_jsonl) {
  done <- load_jsonl_keys(resume_jsonl)
  gh <- git_head()
  cells <- list()
  for (k in spec$kernels) {
    for (regime in k$regimes) {
      clk <- normalise_clock(regime)
      key <- sprintf("%s|%s|%s", gh,
                     if (is.na(clk)) "native" else as.character(clk),
                     k$id)
      already <- key %in% done
      cells[[length(cells) + 1L]] <- list(
        cell_id          = k$id,
        clock_target_mhz = if (is.na(clk)) NULL else clk,
        clock_label      = if (is.na(clk)) "native" else as.character(clk),
        exe              = k$exe,
        args             = k$args,
        match            = k$match,
        section          = k$section,
        value_label      = k$value_label,
        n_samples        = k$n_samples,
        warmup           = k$warmup,
        band_mhz         = k$band_mhz,
        resume_key       = key,
        already_done     = already
      )
    }
  }
  # Sort: native first (no lock churn), then ascending clock, then by cell_id.
  ord <- order(
    vapply(cells, function(c) {
      if (is.null(c$clock_target_mhz)) -1L else c$clock_target_mhz
    }, integer(1)),
    vapply(cells, function(c) c$cell_id, character(1))
  )
  cells <- cells[ord]
  list(
    git_head = gh,
    spec_path = NULL,                # filled by caller
    n_cells = length(cells),
    n_pending = sum(!vapply(cells, function(c) c$already_done, logical(1))),
    cells = cells
  )
}

# ----------------------------------------------------------------------
# Measure
# ----------------------------------------------------------------------
parse_bench_line <- function(out, match, value_label, section) {
  lines <- out
  if (!is.null(section) && nzchar(section)) {
    in_sec <- FALSE
    keep <- logical(length(lines))
    for (i in seq_along(lines)) {
      s <- trimws(lines[[i]])
      is_hdr <- grepl("^(===|---)", s, perl = TRUE)
      if (is_hdr) {
        in_sec <- grepl(section, lines[[i]], fixed = TRUE)
        next
      }
      keep[[i]] <- in_sec
    }
    lines <- lines[keep]
  }
  cand <- grep(match, lines, fixed = TRUE, value = TRUE)
  cand <- cand[grepl("ms", cand, fixed = TRUE)]
  if (!is.null(value_label) && nzchar(value_label))
    cand <- cand[grepl(value_label, cand, fixed = TRUE)]
  if (length(cand) == 0L)
    return(list(ms = NA_real_, throughput = NA_real_, unit = NA_character_))

  line <- cand[[length(cand)]]
  ms <- suppressWarnings(as.numeric(
    sub(".*?([0-9.]+)\\s*ms.*", "\\1", line)))

  if (!is.null(value_label) && nzchar(value_label)) {
    rx <- gsub("([().])", "\\\\\\1", value_label)
    tp <- suppressWarnings(as.numeric(
      sub(sprintf(".*?([0-9.]+)\\s*%s.*", rx), "\\1", line)))
    unit <- if (grepl("TOPS", value_label, ignore.case = TRUE))
              "TOPS" else "GFLOPS"
  } else {
    m <- regmatches(line,
                    regexec("([0-9][0-9,.]*)\\s*(GFLOPS|TOPS)",
                            line, perl = TRUE, ignore.case = TRUE))[[1]]
    if (length(m) >= 3L) {
      tp <- as.numeric(gsub(",", "", m[[2]], fixed = TRUE))
      unit <- toupper(m[[3]])
    } else {
      tp <- NA_real_
      unit <- NA_character_
    }
  }
  list(ms = ms, throughput = tp, unit = unit)
}

throttle_str <- function(post) {
  reasons <- setdiff(post$gpu$throttle, "GpuIdle")
  if (length(reasons) == 0L) "none" else paste(reasons, collapse = ",")
}

static_gpu_info <- function() {
  res <- tryCatch(
    system2("nvidia-smi",
            c("--query-gpu=driver_version,uuid",
              "--format=csv,noheader,nounits"),
            stdout = TRUE, stderr = FALSE),
    error = function(e) character(0))
  if (length(res) == 0L)
    return(list(driver_version = NA_character_, uuid = NA_character_))
  parts <- strsplit(trimws(res[[1]]), ",\\s*")[[1]]
  list(
    driver_version = if (length(parts) >= 1L) trimws(parts[[1]]) else NA_character_,
    uuid           = if (length(parts) >= 2L) trimws(parts[[2]]) else NA_character_
  )
}

run_one_sample <- function(exe_abs, args) {
  pre  <- capture_gpu_state()
  out  <- suppressWarnings(
    system2(exe_abs, as.character(args),
            stdout = TRUE, stderr = TRUE))
  post <- capture_gpu_state()
  st   <- attr(out, "status")
  list(out = out, pre = pre, post = post,
       rc = if (is.null(st)) 0L else as.integer(st))
}

# Append one row as a single JSONL line. `cat(..., append = TRUE)`
# is atomic at line boundaries on POSIX (and on Windows for short
# writes < PIPE_BUF / page size) — a Ctrl+C between rows leaves
# previous rows intact; a Ctrl+C mid-row leaves at most one
# truncated final line that the reader drops.
append_jsonl_row <- function(jsonl_path, row) {
  json <- toJSON(row, auto_unbox = TRUE, na = "null", null = "null",
                 digits = NA)
  cat(json, "\n", sep = "", file = jsonl_path, append = TRUE)
}

measure_cell <- function(cell, jsonl_path, run_id, gh) {
  exe_abs <- path_abs(here(cell$exe))
  if (!file_exists(exe_abs))
    stop(sprintf("exe not found: %s", exe_abs))

  static <- static_gpu_info()

  args <- unlist(cell$args)
  warmup_n  <- as.integer(cell$warmup)
  n_samples <- as.integer(cell$n_samples)
  band      <- as.integer(cell$band_mhz)
  clk_tgt   <- cell$clock_target_mhz  # NULL for native
  clk_lo    <- if (is.null(clk_tgt)) NA_integer_ else clk_tgt - band
  clk_hi    <- if (is.null(clk_tgt)) NA_integer_ else clk_tgt + band

  cli_h2(sprintf("measure %s @ %s MHz", cell$id, cell$clock_label))

  # Warmup — drives the GPU to whatever steady clock the regime
  # permits. No data captured.
  withr_dir <- getwd()
  on.exit(setwd(withr_dir), add = TRUE)
  setwd(path_dir(exe_abs))

  for (i in seq_len(warmup_n)) {
    run_one_sample(exe_abs, args)
  }

  for (i in seq_len(n_samples)) {
    r <- run_one_sample(exe_abs, args)

    # Ctrl+C while R is blocked in system2 sends SIGINT to every
    # process attached to the console: the bench child catches it and
    # exits 130 (128 + SIGINT). R's own SIGINT handler may not fire
    # (the signal was consumed by the child), so R continues to the
    # next iteration. Detect bench rc==130 and propagate as cancel —
    # otherwise the user has to press Ctrl+C once per remaining
    # sample to actually stop. (Field report: needed three presses.)
    if (identical(r$rc, 130L)) {
      message("Bench exited 130 (SIGINT) — treating as user cancel")
      quit(save = "no", status = 130L)
    }

    parsed <- parse_bench_line(r$out, cell$match,
                               cell$value_label, cell$section)
    thr <- throttle_str(r$post)
    clk_obs <- as.integer(r$post$gpu$clock_sm %||% NA_integer_)

    valid <- TRUE
    reject <- NA_character_
    if (r$rc != 0L) { valid <- FALSE; reject <- sprintf("rc=%d", r$rc) }
    else if (is.na(parsed$throughput)) {
      valid <- FALSE; reject <- "parse_failed"
    } else if (thr != "none") {
      valid <- FALSE; reject <- sprintf("throttle:%s", thr)
    } else if (!is.null(clk_tgt) && !is.na(clk_obs)) {
      if (clk_obs < clk_lo || clk_obs > clk_hi) {
        valid <- FALSE
        reject <- sprintf("clock_out_of_band:%d not in [%d,%d]",
                          clk_obs, clk_lo, clk_hi)
      }
    }

    row <- list(
      run_id           = run_id,
      ts_utc           = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z",
                                tz = "UTC"),
      git_head         = gh,
      driver_version   = static$driver_version,
      gpu_uuid         = static$uuid,
      gpu_mode         = r$post$host$gpu_mode %||% NA,
      cell_id          = cell$id,
      exe              = cell$exe,
      args_str         = paste(as.character(args), collapse = ","),
      clock_target_mhz = if (is.null(clk_tgt)) NA_integer_ else clk_tgt,
      clock_observed_mhz = clk_obs,
      clock_mem_mhz    = as.integer(r$post$gpu$clock_mem %||% NA_integer_),
      sample_idx       = i,
      ms               = parsed$ms,
      throughput       = parsed$throughput,
      unit             = parsed$unit,
      power_w          = as.numeric(r$post$gpu$power_w %||% NA_real_),
      temp_c           = as.numeric(r$post$gpu$temp_c %||% NA_real_),
      throttle_str     = thr,
      valid            = valid,
      reject_reason    = reject,
      rc               = r$rc
    )
    append_jsonl_row(jsonl_path, row)

    cli_alert(sprintf(
      "%s sample %d/%d : %s %s   %s ms   clk %s MHz   %s",
      cell$id, i, n_samples,
      if (is.na(parsed$throughput)) "NA"
        else format(round(parsed$throughput, 0), big.mark = ""),
      if (is.na(parsed$unit)) "" else parsed$unit,
      if (is.na(parsed$ms)) "NA" else round(parsed$ms, 3),
      if (is.na(clk_obs)) "NA" else clk_obs,
      if (valid) cli::col_green("OK") else cli::col_red(sprintf("REJECT(%s)", reject))
    ))
  }
}

# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.null(args$mode))
    stop("--mode plan|measure required")

  if (args$mode == "plan") {
    if (is.null(args$spec)) stop("--spec required for plan")
    spec <- load_spec(args$spec)
    plan <- build_plan(spec, args$resume_jsonl)
    plan$spec_path <- normalizePath(args$spec, mustWork = TRUE)
    json <- toJSON(plan, auto_unbox = TRUE, na = "null", null = "null",
                   pretty = TRUE)
    if (!is.null(args$out)) {
      # JSON goes to file; stdout stays free for renv NOTEs and other
      # log noise. The orchestrator reads --out and discards stdout.
      writeLines(json, args$out)
      cat(sprintf("plan written to %s (n_cells=%d, n_pending=%d)\n",
                  args$out, plan$n_cells, plan$n_pending))
    } else {
      cat(json, "\n", sep = "")
    }
    return(invisible())
  }

  if (args$mode == "measure") {
    for (req in c("spec", "cell_id", "clock_target", "jsonl"))
      if (is.null(args[[req]]))
        stop(sprintf("--%s required for measure",
                     gsub("_", "-", req)))
    spec <- load_spec(args$spec)
    cell <- NULL
    for (k in spec$kernels) if (k$id == args$cell_id) { cell <- k; break }
    if (is.null(cell)) stop(sprintf("cell_id not in spec: %s", args$cell_id))

    clk <- normalise_clock(args$clock_target)
    cell$clock_target_mhz <- if (is.na(clk)) NULL else clk
    cell$clock_label      <- if (is.na(clk)) "native" else as.character(clk)

    if (is.null(args$run_id))
      args$run_id <- format(Sys.time(), "%Y%m%dT%H%M%S")
    gh <- git_head()
    measure_cell(cell, args$jsonl, args$run_id, gh)
    return(invisible())
  }

  stop(sprintf("unknown mode: %s", args$mode))
}

if (sys.nframe() == 0L) {
  # Trap Ctrl+C so the exit code distinguishes "user cancelled" (130)
  # from "real failure" (1). PowerShell orchestrator checks for 130
  # to abort the sweep instead of continuing to the next cell.
  tryCatch(
    main(),
    interrupt = function(c) {
      message("Interrupted by user (SIGINT)")
      quit(save = "no", status = 130)
    }
  )
}

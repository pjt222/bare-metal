#!/usr/bin/env Rscript
# bench_all.R - one-click full-corpus benchmark runner (issue #124).
#
# Runs EVERY bench executable in the kernel corpus and records every
# result with full metadata. The guiding principle (docs/benchmark_
# methodology.md, "Methodology for a full run everything pass") is
# SKIP NOTHING, RECORD EVERYTHING:
#
#   - Discover the whole corpus the same way the Makefile builds it
#     ($(BENCH_EXES): every bench.cu + kernels/**/bench_*.cu).
#   - Per measurable config, retry up to --max-attempts until --min-valid
#     clean samples are collected; classify each attempt from its pre/post
#     capture_gpu_state() snapshot + exit code; cool down (adaptive)
#     between attempts.
#   - Never abort the corpus: a config that cannot reach --min-valid is
#     `degraded`; one that fails every attempt is `failed`; a missing exe
#     is `not-built`. The runner continues regardless.
#   - Output keeps EVERY attempt (samples.jsonl + results.json) plus a
#     per-config summary, so "no failures" is a property the reader
#     verifies from a complete report, not a guarantee baked into the run.
#
# This is the on-demand "collect everything" pass. `make bench` /
# bench_regress.R stays the fast regression gate (5 baselined kernels).
# For clock/power context around a run, see scripts/probe/probe_gpu_power.R.
#
# Measurement reuses the cuasmR package API (issue #134): run_bench,
# parse_throughput, capture_gpu_state, classify_meta, validate_sample,
# collect_valid_samples, report_median_metrics, append_jsonl_row. Nothing
# measurement-related is reimplemented here.
#
# NOT every bench in the corpus emits a single parseable throughput
# number. The spec (scripts/bench/bench_all.yml) tags each exe:
#   measurable: false  -> A/B sweep table / ms-only / correctness harness:
#                         run ONCE to confirm it executes, status
#                         `non-measurable` (a parse miss here is EXPECTED,
#                         never a `failed`).
#   run: false         -> cannot run meaningfully here (stub library,
#                         needs external data): documented `skipped`.
# An exe with NO spec entry is still run, with --default-args + a generic
# parse, flagged spec_source="default" and reported in a SEPARATE bucket --
# a `failed` there means "the default args were probably wrong", NOT a real
# kernel failure. Never read the buckets as the same kind of failure.
#
# Usage:
#   Rscript scripts/bench/bench_all.R                  # full corpus
#   Rscript scripts/bench/bench_all.R --list           # plan only, no GPU
#   Rscript scripts/bench/bench_all.R --only hgemm_16warp_2048
#   Rscript scripts/bench/bench_all.R --min-valid 3 --max-attempts 10

suppressPackageStartupMessages({
  library(jsonlite)
  library(yaml)
})

# GPU + host state, run, parse, validate, collect, median, JSONL: all
# from cuasmR (issue #134). LD_LIBRARY_PATH WSL guard is in cuasmR .onLoad.
suppressMessages(library(cuasmR))

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

# ----------------------------------------------------------------------
# Repo root: walk up from the script dir until a .git / renv.lock marker
# (same resilient resolver as bench_regress.R).
# ----------------------------------------------------------------------
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

DEFAULT_SPEC   <- file.path(REPO_ROOT, "scripts", "bench", "bench_all.yml")
BASELINES_PATH <- file.path(REPO_ROOT, "data", "baselines.json")

# Project-wide default fairness gate, shared with the regression gate:
# reject any sample whose pre/post GPU state shows a non-idle throttle.
# Read from baselines.json so bench_all and bench_regress agree.
DEFAULT_VALID_WHEN <- local({
  vw <- list(require_no_throttle = TRUE, allow_throttle = c("GpuIdle"))
  if (file.exists(BASELINES_PATH)) {
    b <- tryCatch(jsonlite::fromJSON(BASELINES_PATH, simplifyVector = FALSE),
                  error = function(e) NULL)
    if (!is.null(b$default_valid_when)) {
      dvw <- b$default_valid_when
      dvw$comment <- NULL
      vw <- dvw
    }
  }
  vw
})

# ======================================================================
# Pure functions (GPU-free, unit-tested in tests/bench_all/).
# ======================================================================

#' Discover the full bench corpus exactly as the Makefile's $(BENCH_EXES):
#' every `bench.cu` under the repo (excluding tools/, experiments/, renv/,
#' .git/) + every `kernels/**/bench_*.cu`. Returns repo-relative .cu
#' source paths, sorted+unique. Shell-free so the test runs on any box.
discover_corpus <- function(root) {
  prune <- c("tools", "experiments", "renv", ".git")
  all_cu <- list.files(root, pattern = "\\.cu$", recursive = TRUE,
                       full.names = FALSE)
  if (!length(all_cu)) return(character(0))
  top <- vapply(strsplit(all_cu, "/", fixed = TRUE), `[`, character(1), 1L)
  all_cu <- all_cu[!(top %in% prune)]
  base <- basename(all_cu)
  is_bench   <- base == "bench.cu"
  is_variant <- grepl("^bench_.*\\.cu$", base) & startsWith(all_cu, "kernels/")
  sort(unique(all_cu[is_bench | is_variant]))
}

#' exe path for a bench .cu source (strip the .cu).
exe_for_src <- function(src) sub("\\.cu$", "", src)

#' Auto id for a corpus exe with no spec entry. e.g.
#' "kernels/attention/cross_attention/bench_v2" ->
#' "attention_cross_attention_bench_v2".
auto_id <- function(exe) gsub("[/]", "_", sub("^kernels/", "", exe))

#' Merge the discovered corpus with the YAML spec.
#'
#' Every spec config becomes a config record (spec_source="known"). Every
#' corpus exe NOT referenced by any spec config becomes one default config
#' (spec_source="default", args=default_args, generic parse). Result: EVERY
#' exe is covered at least once, nothing dropped.
#'
#' @return list of config records (id, exe, src, args, match, section,
#'   value_label, unit, valid_when, n_samples, timeout, run, measurable,
#'   spec_source, in_corpus, notes).
merge_spec <- function(corpus_src, spec_kernels, default_args) {
  corpus_exes <- vapply(corpus_src, exe_for_src, character(1))
  src_by_exe  <- stats::setNames(corpus_src, corpus_exes)

  mk <- function(k, spec_source) {
    exe <- k$exe
    in_corpus <- exe %in% corpus_exes
    list(
      id          = k$id %||% auto_id(exe),
      exe         = exe,
      src         = if (in_corpus) src_by_exe[[exe]] else paste0(exe, ".cu"),
      args        = as.character(unlist(k$args %||% list())),
      match       = k$match %||% NULL,
      section     = k$section %||% NULL,
      value_label = k$value_label %||% NULL,
      unit        = k$unit %||% NA_character_,
      valid_when  = k$valid_when %||% NULL,
      n_samples   = k$n_samples %||% NULL,
      timeout     = k$timeout %||% NULL,
      run         = if (identical(k$run, FALSE)) FALSE else TRUE,
      measurable  = if (identical(k$measurable, FALSE)) FALSE else TRUE,
      verified    = isTRUE(k$verified),
      spec_source = spec_source,
      in_corpus   = in_corpus,
      notes       = k$note %||% k$notes %||% ""
    )
  }

  configs <- list()
  specced_exes <- character(0)
  for (k in spec_kernels %||% list()) {
    specced_exes <- c(specced_exes, k$exe)
    configs[[length(configs) + 1L]] <- mk(k, "known")
  }

  uncovered <- setdiff(corpus_exes, unique(specced_exes))
  for (exe in sort(uncovered)) {
    configs[[length(configs) + 1L]] <- mk(list(
      exe = exe, args = as.list(default_args),
      note = "no spec entry; default args + generic parse (invocation UNVERIFIED)"
    ), "default")
  }
  configs
}

#' Status for a finished MEASURABLE config (not-built / skipped /
#' non-measurable are decided by the caller). complete -> ok; some but
#' not enough -> degraded; none -> failed.
classify_status <- function(complete, n_valid_collected) {
  if (isTRUE(complete)) "ok"
  else if (n_valid_collected > 0L) "degraded"
  else "failed"
}

#' Canonical bucket for a per-sample reject reason (for the histogram).
reason_bucket <- function(reason) {
  if (is.null(reason) || length(reason) == 0L || is.na(reason)) return("unknown")
  if      (startsWith(reason, "crash"))       "crash"
  else if (startsWith(reason, "parse-fail"))  "parse-fail"
  else if (startsWith(reason, "unfair"))      "unfair"
  else if (startsWith(reason, "no-gpu-meta")) "no-gpu-meta"
  else if (startsWith(reason, "clock"))       "clock-band"
  else if (startsWith(reason, "error"))       "error"
  else                                        "other"
}

#' Histogram of reject reasons -> named integer vector (counts by bucket).
reject_histogram <- function(reasons) {
  if (!length(reasons)) return(stats::setNames(integer(0), character(0)))
  buckets <- vapply(reasons, reason_bucket, character(1))
  tb <- table(buckets)
  stats::setNames(as.integer(tb), names(tb))
}

#' Short "top reason" string for a degraded/failed config.
top_reject <- function(hist) {
  if (!length(hist)) return("")
  ord <- order(hist, decreasing = TRUE)
  paste(sprintf("%s:%d", names(hist)[ord], hist[ord]), collapse = " ")
}

#' Pick the reported unit: spec unit wins (authoritative, e.g. GB/s, which
#' parse_throughput would mislabel GFLOPS), else the parsed unit.
pick_unit <- function(spec_unit, parsed_units) {
  if (!is.null(spec_unit) && length(spec_unit) && !is.na(spec_unit) &&
      nzchar(spec_unit)) return(spec_unit)
  u <- parsed_units[!is.na(parsed_units)]
  if (length(u)) u[[1]] else NA_character_
}

#' Build the per-config summary from valid samples + every attempt. Pure:
#' the test drives it with synthetic input.
summarise_config <- function(cfg, valid_tputs, valid_mss, valid_units,
                             reject_reasons, n_attempts, complete,
                             attempts = list()) {
  n_valid <- length(valid_tputs)
  status  <- classify_status(complete, n_valid)
  hist    <- reject_histogram(reject_reasons)
  med     <- if (n_valid > 0L) report_median_metrics(valid_tputs, valid_mss)
             else NULL
  list(
    id                  = cfg$id,
    exe                 = cfg$exe,
    src                 = cfg$src,
    args                = cfg$args,
    spec_source         = cfg$spec_source,
    invocation_verified = identical(cfg$spec_source, "known"),
    measurable          = isTRUE(cfg$measurable),
    verified            = isTRUE(cfg$verified),
    status              = status,
    n_valid             = n_valid,
    n_attempts          = as.integer(n_attempts),
    median_throughput   = med$median_throughput %||% NA_real_,
    median_ms           = med$median_ms %||% NA_real_,
    tput_lo             = med$tput_lo %||% NA_real_,
    tput_hi             = med$tput_hi %||% NA_real_,
    unit                = pick_unit(cfg$unit, valid_units),
    reject_buckets      = as.list(hist),
    top_reject          = top_reject(hist),
    notes               = cfg$notes,
    attempts            = attempts
  )
}

#' A config skeleton summary for the non-running cases (not-built /
#' skipped / non-measurable), so the report shape is uniform.
skeleton_summary <- function(cfg, status, n_valid, n_attempts, note,
                             attempts = list(), median = NA_real_,
                             median_ms = NA_real_, unit = NA_character_) {
  list(
    id = cfg$id, exe = cfg$exe, src = cfg$src, args = cfg$args,
    spec_source = cfg$spec_source,
    invocation_verified = identical(cfg$spec_source, "known"),
    measurable = isTRUE(cfg$measurable),
    verified = isTRUE(cfg$verified),
    status = status, n_valid = as.integer(n_valid),
    n_attempts = as.integer(n_attempts),
    median_throughput = median, median_ms = median_ms,
    tput_lo = NA_real_, tput_hi = NA_real_,
    unit = pick_unit(cfg$unit, character(0)),
    reject_buckets = list(), top_reject = note,
    notes = cfg$notes, attempts = attempts
  )
}

#' Render the human-readable summary.md. Pure string builder -- the test
#' asserts the measurable / non-measurable / default-args buckets stay
#' separate (the advisor invariant: a default-args or non-measurable parse
#' miss must never read as a real kernel failure).
render_summary_md <- function(run_meta, summaries) {
  fmt <- function(x) if (is.na(x)) "-" else format(round(x, 1), big.mark = "", nsmall = 1)
  row <- function(s) {
    spread <- if (is.na(s$tput_lo)) "-" else sprintf("%s-%s", fmt(s$tput_lo), fmt(s$tput_hi))
    ver <- if (isTRUE(s$verified)) "verified" else if (isTRUE(s$measurable)) "infer" else "-"
    sprintf("| %s | `%s` | %s | %s | %d/%d | %s | %s | %s | %s |",
            s$id, paste(s$args, collapse = " "), s$status, ver,
            s$n_valid, s$n_attempts, fmt(s$median_throughput),
            if (is.na(s$unit)) "" else s$unit, spread, s$top_reject)
  }
  hdr <- paste0(
    "| id | args | status | spec | valid/try | median | unit | spread | note |\n",
    "|----|------|--------|------|-----------|--------|------|--------|------|")
  counts <- function(ss) {
    st <- vapply(ss, function(s) s$status, character(1))
    paste(sprintf("%s=%d", names(table(st)), as.integer(table(st))), collapse = " ")
  }
  is_measurable_known <- function(s) identical(s$spec_source, "known") && isTRUE(s$measurable)
  is_nonmeasurable    <- function(s) identical(s$spec_source, "known") && !isTRUE(s$measurable)
  is_default          <- function(s) identical(s$spec_source, "default")

  perf    <- Filter(is_measurable_known, summaries)
  nonmeas <- Filter(is_nonmeasurable,    summaries)
  deflt   <- Filter(is_default,          summaries)

  out <- c(
    "# bench-all full-corpus report",
    "",
    sprintf("- Generated: %s", run_meta$ts_utc),
    sprintf("- Commit: %s%s", substr(run_meta$git_head %||% "?", 1, 12),
            if (isTRUE(run_meta$git_dirty)) " (dirty)" else ""),
    sprintf("- Host: %s", run_meta$host %||% "?"),
    sprintf("- GPU: %s (driver %s, %s, %s)", run_meta$gpu_name %||% "?",
            run_meta$driver_version %||% "?", run_meta$sm_arch %||% "?",
            run_meta$nvcc %||% "?"),
    sprintf("- GPU mode: %s   |   clock: %s", run_meta$gpu_mode %||% "?",
            run_meta$clock_lock %||% "native"),
    "",
    "Native-boost run. Power-bound kernels (e.g. igemm_sparse 4096) throttle",
    "here and land `degraded`/`failed` -- expected and recorded, not hidden.",
    "For their fair number use a host-side clock lock",
    "(scripts/probe/run_locked_eval.ps1); for clock/power context see",
    "scripts/probe/probe_gpu_power.R.",
    "",
    "## Measurable corpus (single-number perf)",
    "",
    "`spec=verified`: args + parse hint confirmed by a recorded baseline /",
    "sweep number. `spec=infer`: hint read from the bench source -- this run",
    "confirms it. A `failed`/`parse-fail` on an `infer` row is most likely a",
    "spec-hint bug to fix in bench_all.yml, NOT a real kernel failure.",
    sprintf("_%s_", counts(perf)), "", hdr,
    vapply(perf, row, character(1)), "",
    "## Non-measurable / skipped (documented, NOT failures)",
    "",
    "A/B sweep tables, ms-only composed pipelines, correctness harnesses,",
    "and benches needing external data / unavailable libs. `non-measurable`",
    "means it ran but emits no single number; `skipped` means it was not run",
    "(reason in the note). Neither is a kernel failure.",
    sprintf("_%s_", counts(nonmeas)), "", hdr,
    vapply(nonmeas, row, character(1)), "",
    "## Discovered without a spec (default args -- invocation UNVERIFIED)",
    "",
    "Empty unless a new bench was added to the corpus without a",
    "`bench_all.yml` entry. A `failed` here most likely means the default",
    "args were wrong, NOT a real kernel failure -- add a spec entry.",
    sprintf("_%s_", if (length(deflt)) counts(deflt) else "none"), "", hdr,
    vapply(deflt, row, character(1)), ""
  )
  paste(out, collapse = "\n")
}

# ======================================================================
# GPU glue (not unit-tested; needs a real GPU + built corpus).
# ======================================================================

git_head <- function() {
  res <- tryCatch(system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE),
                  error = function(e) NA_character_)
  if (length(res) == 0L) NA_character_ else res[[1]]
}
git_dirty <- function() {
  res <- tryCatch(system2("git", c("status", "--porcelain"), stdout = TRUE, stderr = FALSE),
                  error = function(e) character(0))
  length(res) > 0L
}
nvcc_release <- function() {
  tryCatch({
    r <- system2("nvcc", "--version", stdout = TRUE, stderr = FALSE)
    m <- regmatches(paste(r, collapse = " "),
                    regexec("release\\s+([0-9.]+)", paste(r, collapse = " "), perl = TRUE))[[1]]
    if (length(m) >= 2) paste0("CUDA ", m[2]) else NA_character_
  }, error = function(e) NA_character_)
}
smi_static <- function(field) {
  tryCatch({
    r <- system2("nvidia-smi", c(sprintf("--query-gpu=%s", field),
                                 "--format=csv,noheader,nounits"),
                 stdout = TRUE, stderr = FALSE)
    trimws(r[[1]])
  }, error = function(e) NA_character_)
}

build_run_meta <- function() {
  list(
    ts_utc         = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
    git_head       = git_head(),
    git_dirty      = git_dirty(),
    host           = unname(Sys.info()[["nodename"]]),
    os             = tryCatch(readLines("/proc/version", n = 1, warn = FALSE),
                              error = function(e) NA_character_),
    nvcc           = nvcc_release(),
    driver_version = smi_static("driver_version"),
    gpu_name       = smi_static("name"),
    gpu_memory_mb  = smi_static("memory.total"),
    gpu_mode       = tolower(Sys.getenv("BARE_METAL_GPU_MODE", unset = "unknown")),
    clock_lock     = "native",
    sm_arch        = "sm_86"
  )
}

#' Build a per-attempt JSONL row (also kept in results.json attempts[]).
attempt_row <- function(cfg, s, ok, reason, attempt, run_id, gh) {
  post <- s$r$post
  list(
    run_id = run_id, ts_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
    git_head = gh, cell_id = cfg$id, exe = cfg$exe, spec_source = cfg$spec_source,
    args_str = paste(cfg$args, collapse = ","), attempt = as.integer(attempt),
    ms = s$parsed$ms, throughput = s$parsed$throughput, unit = s$parsed$unit,
    clock_sm_mhz = as.integer((post$gpu$clock_sm %||% NA_integer_)),
    power_w = as.numeric((post$gpu$power_w %||% NA_real_)),
    temp_c = as.numeric((post$gpu$temp_c %||% NA_real_)),
    throttle = paste(setdiff(post$gpu$throttle %||% character(0), "GpuIdle"), collapse = ","),
    gpu_mode = post$host$gpu_mode %||% NA_character_,
    valid = isTRUE(ok), reject_reason = reason %||% NA_character_, rc = s$r$rc
  )
}

one_run <- function(exe_abs, cfg, timeout) {
  r <- run_bench(exe_abs, cfg$args, timeout = timeout)
  parsed <- parse_throughput(r$out, match = cfg$match, section = cfg$section,
                             value_label = cfg$value_label, pick = "first")
  list(r = r, parsed = parsed)
}

#' Measure one config: dispatch on run / measurable, never throw.
measure_config <- function(cfg, opts, jsonl_path, run_id, gh) {
  if (identical(cfg$run, FALSE))
    return(skeleton_summary(cfg, "skipped", 0L, 0L,
                            cfg$notes %||% "run: false in spec"))

  exe_abs <- file.path(REPO_ROOT, cfg$exe)
  if (!file.exists(exe_abs))
    return(skeleton_summary(cfg, "not-built", 0L, 0L, "exe not built (try: make all)"))
  exe_abs <- normalizePath(exe_abs, mustWork = TRUE)

  prev_wd <- getwd(); setwd(dirname(exe_abs)); on.exit(setwd(prev_wd), add = TRUE)
  timeout <- as.integer(cfg$timeout %||% opts$timeout)

  # Non-measurable: run ONCE to confirm it executes. A parse miss is
  # EXPECTED (no single number) -> status non-measurable; a non-zero exit
  # is a real run failure -> status failed.
  if (identical(cfg$measurable, FALSE)) {
    # Non-measurable benches often run a long internal sweep (many shapes).
    # Give the single confirming run a more generous timeout so a slow-but-
    # fine sweep is not killed and mislabelled `failed`.
    nm_timeout <- as.integer(cfg$timeout %||% max(opts$timeout, 300L))
    s <- one_run(exe_abs, cfg, nm_timeout)
    if (identical(s$r$rc, 130L)) { message("SIGINT -- cancelling"); quit(save = "no", status = 130L) }
    ok <- identical(s$r$rc, 0L)
    row <- attempt_row(cfg, s, ok, if (ok) "non-measurable" else sprintf("crash(exit=%d)", s$r$rc),
                       1L, run_id, gh)
    append_jsonl_row(jsonl_path, row)
    note <- if (ok) (cfg$notes %||% "ran; no single-number metric")
            else sprintf("ran but exited %d", s$r$rc)
    return(skeleton_summary(cfg, if (ok) "non-measurable" else "failed",
                            0L, 1L, note, attempts = list(row)))
  }

  vw      <- cfg$valid_when %||% DEFAULT_VALID_WHEN
  n_valid <- as.integer(cfg$n_samples %||% opts$min_valid)

  # `attempts` accumulates every attempt row. on_sample's `attempts[[..]] <<- `
  # binds THIS frame because collect_valid_samples invokes on_sample
  # synchronously, inline, before measure_config returns (bench_measure.R). If
  # that ever becomes async/deferred the super-assignment would break -- keep it
  # synchronous, or pass `attempts` through explicitly.
  attempts <- list()

  on_sample <- function(attempt, ok, s, reason) {
    if (identical(s$r$rc, 130L)) { message("SIGINT -- cancelling"); quit(save = "no", status = 130L) }
    row <- attempt_row(cfg, s, ok, reason, attempt, run_id, gh)
    append_jsonl_row(jsonl_path, row)
    attempts[[length(attempts) + 1L]] <<- row
    cat(sprintf("    %-26s try %2d  %s %s  %s\n", cfg$id, attempt,
                if (is.na(s$parsed$throughput)) "NA" else format(round(s$parsed$throughput, 0), big.mark = ""),
                if (is.na(s$parsed$unit)) "" else s$parsed$unit,
                if (isTRUE(ok)) "OK" else sprintf("REJECT(%s)", reason)))
    # Adaptive cooldown: longer after an unfair/throttle reject so the GPU
    # sheds heat / leaves the power cap before the next attempt.
    cool <- opts$cooldown
    if (!isTRUE(ok) && !is.na(reason) && startsWith(reason, "unfair"))
      cool <- cool * opts$cooldown_throttle_mult
    if (cool > 0) Sys.sleep(cool)
  }

  res <- collect_valid_samples(
    sample_fn = function() one_run(exe_abs, cfg, timeout),
    validate_fn = function(s) validate_sample(s$r$rc, s$parsed$throughput,
                                              s$r$pre, s$r$post, valid_when = vw),
    n_valid = n_valid, max_attempts = opts$max_attempts, on_sample = on_sample)

  tputs <- vapply(res$samples, function(s) s$parsed$throughput, numeric(1))
  mss   <- vapply(res$samples, function(s) s$parsed$ms %||% NA_real_, numeric(1))
  units <- vapply(res$samples, function(s) s$parsed$unit %||% NA_character_, character(1))
  summarise_config(cfg, tputs, mss, units, res$rejected, res$attempts,
                   res$complete, attempts = attempts)
}

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list(spec = DEFAULT_SPEC, out_dir = NULL, min_valid = 5L,
              max_attempts = 15L, cooldown = 2, cooldown_throttle_mult = 4,
              timeout = 120L, only = NULL, default_args = character(0),
              list_only = FALSE)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if      (a == "--spec")         { out$spec <- argv[i+1]; i <- i+2 }
    else if (a == "--out-dir")      { out$out_dir <- argv[i+1]; i <- i+2 }
    else if (a == "--min-valid")    { out$min_valid <- as.integer(argv[i+1]); i <- i+2 }
    else if (a == "--max-attempts") { out$max_attempts <- as.integer(argv[i+1]); i <- i+2 }
    else if (a == "--cooldown")     { out$cooldown <- as.numeric(argv[i+1]); i <- i+2 }
    else if (a == "--timeout")      { out$timeout <- as.integer(argv[i+1]); i <- i+2 }
    else if (a == "--only")         { out$only <- argv[i+1]; i <- i+2 }
    else if (a == "--default-args") { out$default_args <- strsplit(argv[i+1], ",", fixed=TRUE)[[1]]; i <- i+2 }
    else if (a %in% c("--list", "--dry-run")) { out$list_only <- TRUE; i <- i+1 }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: bench_all.R [--spec F] [--out-dir D] [--min-valid N]",
          "[--max-attempts N]\n",
          "                   [--cooldown S] [--timeout S] [--only ID]",
          "[--default-args a,b,c] [--list]\n",
          "  --list  print the planned corpus (spec_source per exe) and exit (no GPU)\n",
          sep = "")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  out
}

load_spec_kernels <- function(spec_path) {
  if (!file.exists(spec_path)) {
    cat(sprintf("WARN: spec not found (%s); every bench runs default-args.\n", spec_path))
    return(list())
  }
  (yaml::read_yaml(spec_path))$kernels %||% list()
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))

  corpus  <- discover_corpus(REPO_ROOT)
  spec_k  <- load_spec_kernels(opts$spec)
  configs <- merge_spec(corpus, spec_k, opts$default_args)
  if (!is.null(opts$only))
    configs <- Filter(function(c) identical(c$id, opts$only), configs)
  if (!length(configs)) { cat("No configs to run.\n"); quit(status = 1) }

  n_known   <- sum(vapply(configs, function(c) c$spec_source == "known", logical(1)))
  n_default <- length(configs) - n_known
  n_meas    <- sum(vapply(configs, function(c) isTRUE(c$measurable) && !identical(c$run, FALSE), logical(1)))

  cat(strrep("=", 72), "\n", sep = "")
  cat("  bench-all -- full-corpus run (skip nothing, record everything)\n")
  cat(sprintf("  corpus %d exes | configs %d (known %d, default %d) | measurable %d\n",
              length(corpus), length(configs), n_known, n_default, n_meas))
  cat(sprintf("  min-valid %d | max-attempts %d | cooldown %.1fs\n",
              opts$min_valid, opts$max_attempts, opts$cooldown))
  cat(strrep("=", 72), "\n", sep = "")

  if (opts$list_only) {
    for (c in configs) {
      tag <- if (identical(c$run, FALSE)) "skip" else if (!isTRUE(c$measurable)) "nomeas" else "perf"
      cat(sprintf("  [%-7s %-6s] %-30s  %s  args=[%s]\n",
                  c$spec_source, tag, c$id, c$exe, paste(c$args, collapse = " ")))
    }
    cat(sprintf("\n%d configs planned (%d known, %d default, %d measurable).\n",
                length(configs), n_known, n_default, n_meas))
    quit(status = 0)
  }

  run_id  <- format(Sys.time(), "%Y%m%dT%H%M%S")
  out_dir <- opts$out_dir %||% file.path(REPO_ROOT, "results", "bench_all", run_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  jsonl_path <- file.path(out_dir, "samples.jsonl")
  results_path <- file.path(out_dir, "results.json")
  summary_path <- file.path(out_dir, "summary.md")

  run_meta <- build_run_meta()
  gh <- run_meta$git_head
  .gs <- capture_gpu_state()
  if (!is.null(.gs)) {
    cat(sprintf("  GPU state: %s\n", summarise_meta(.gs, .gs)))
    cat(strrep("=", 72), "\n", sep = "")
  } else {
    cat("  WARNING: no GPU metadata (nvidia-smi absent) -- runs unguarded.\n")
  }

  summaries <- list()
  for (cfg in configs) {
    cat(sprintf("\n[%s] %s  (args=[%s], %s)\n", cfg$id, cfg$exe,
                paste(cfg$args, collapse = " "), cfg$spec_source))
    s <- tryCatch(measure_config(cfg, opts, jsonl_path, run_id, gh),
      error = function(e) {
        cat(sprintf("    ERROR (recorded, corpus continues): %s\n", conditionMessage(e)))
        skeleton_summary(cfg, "failed", 0L, 0L,
                         sprintf("error: %s", conditionMessage(e)))
      })
    summaries[[length(summaries) + 1L]] <- s
    cat(sprintf("    => %s  (%d/%d valid)\n", s$status, s$n_valid, s$n_attempts))
  }

  writeLines(jsonlite::toJSON(list(run_meta = run_meta, configs = summaries),
                              auto_unbox = TRUE, na = "null", null = "null", pretty = TRUE),
             results_path)
  writeLines(render_summary_md(run_meta, summaries), summary_path)

  st <- vapply(summaries, function(s) s$status, character(1))
  cat("\n", strrep("=", 72), "\n", sep = "")
  cat(sprintf("  Done. %s\n",
              paste(sprintf("%s=%d", names(table(st)), as.integer(table(st))), collapse = " ")))
  cat(sprintf("  results.json : %s\n", results_path))
  cat(sprintf("  summary.md   : %s\n", summary_path))
  cat(sprintf("  samples.jsonl: %s\n", jsonl_path))
  cat(strrep("=", 72), "\n", sep = "")

  # bench-all is a data-collection pass, not a gate: exit 0 even with
  # failures (they are recorded in the report; the reader is the judge).
  quit(status = 0)
}

if (sys.nframe() == 0L) {
  tryCatch(main(),
    interrupt = function(c) { message("Interrupted by user (SIGINT)"); quit(save = "no", status = 130) })
}

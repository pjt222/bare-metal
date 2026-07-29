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
# REGIMES (#152 Phase 2). Each config also carries a set of clock
# `regimes` -- "native" (no host-side lock) and/or integer MHz targets an
# elevated `nvidia-smi.exe -lgc` is holding. A config that declares none
# plans `[native]`, and a `measurable: false` / `run: false` config runs
# ONCE-NATIVE whatever it declares (a clock band can only gate a
# parseable number). The planner, the store columns and the collector are
# regime-aware; APPLYING a lock is deliberately not R's job -- the lock
# lifecycle lives in the pwsh orchestrator, outside R, which is what makes
# a Ctrl+C mid-retry safe by construction. Until Phase 3 lands `--regime`,
# this runner measures the native regime and reports the rest as deferred.
#
# Usage:
#   Rscript scripts/bench/bench_all.R                  # full corpus, native
#   Rscript scripts/bench/bench_all.R --list           # cell plan, no GPU
#   Rscript scripts/bench/bench_all.R --only hgemm_16warp_2048
#   Rscript scripts/bench/bench_all.R --min-valid 3 --max-attempts 10
#   Rscript scripts/bench/bench_all_collect.R --print  # JSONL -> RDS rollup

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

# Sample target when neither --min-valid nor the spec's defaults$n_samples
# says otherwise. This is the pre-#152 built-in.
MIN_VALID_FALLBACK <- 5L

# The cell a caller that predates #152 implies. Native, no band -- which
# is the pre-#152 behaviour exactly.
NATIVE_CELL <- list(regime = "native", clock_target_mhz = NA_integer_,
                    band_lo = NA_integer_, band_hi = NA_integer_)

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

# ----------------------------------------------------------------------
# Regimes (#152 Phase 2). A *regime* is the clock the measurement runs
# at: the string "native" (no host-side lock) or an integer MHz that an
# elevated `nvidia-smi.exe -lgc M,M` is holding. R never applies or
# releases a lock -- that lifecycle is owned by the pwsh orchestrator
# outside R (design Q2). R only records which regime it was told it is
# in and gates samples against the matching clock band.
# ----------------------------------------------------------------------

#' Normalise one regime token to an integer clock target.
#'
#' "native" (or NULL/NA) -> NA_integer_; an int-ish MHz in [100,5000] ->
#' integer. Anything else is a spec bug and stops. Same contract as
#' grid_measure.R's normalise_clock so the two specs can merge (Q1).
normalise_clock <- function(x) {
  if (is.null(x) || length(x) == 0L) return(NA_integer_)
  if (length(x) > 1L) stop("normalise_clock: expected one value, got ", length(x))
  if (is.character(x) && identical(tolower(x), "native")) return(NA_integer_)
  if (is.na(x)) return(NA_integer_)
  xi <- suppressWarnings(as.integer(x))
  if (is.na(xi) || xi < 100L || xi > 5000L)
    stop(sprintf("invalid regime/clock value: %s", as.character(x)))
  xi
}

#' Printable label for a clock target. NA -> "native".
regime_label <- function(clock_target_mhz) {
  if (is.na(clock_target_mhz)) "native" else as.character(clock_target_mhz)
}

#' Two-sided clock band for a regime, or NULL for native.
#'
#' Native deliberately gets NULL, not a wide band: there is no clock to
#' hold it to, and `validate_sample` treats NULL as "no band check" --
#' preserving today's native behaviour exactly (design Q5).
#'
#' A LOCKED regime with no usable band is a spec error, not a
#' degrade-to-no-check: silently dropping the band records an unchecked
#' number under a locked key, which is exactly the leak the two-sided
#' band exists to prevent (design Q4).
clock_band_for <- function(clock_target_mhz, band_mhz) {
  if (is.na(clock_target_mhz)) return(NULL)
  b <- suppressWarnings(as.integer(band_mhz))
  if (length(b) != 1L || is.na(b) || b <= 0L)
    stop(sprintf(paste0("locked regime %d MHz needs a positive integer band_mhz, got: %s. ",
                        "A locked cell must never run without a clock-band gate."),
                 clock_target_mhz,
                 if (is.null(band_mhz)) "NULL" else paste(as.character(band_mhz), collapse = ",")))
  c(clock_target_mhz - b, clock_target_mhz + b)
}

#' The clock band a planned cell measures under, or NULL for native.
#'
#' Single source of truth for the band, shared by the planner and by
#' measure_config. measure_config MUST NOT re-derive this from
#' band_lo/band_hi independently -- a second derivation is a second thing
#' that can silently disagree with the recorded row.
cell_clock_band <- function(cell) {
  if (is.null(cell$band_lo) || is.null(cell$band_hi) ||
      is.na(cell$band_lo) || is.na(cell$band_hi)) return(NULL)
  c(cell$band_lo, cell$band_hi)
}

#' The taxonomy x regime rule (design Q1).
#'
#' A measurable, runnable config uses its declared `regimes`, defaulting
#' to `[native]` when it declares none. Everything else -- `run: false`
#' or `measurable: false` -- runs ONCE-NATIVE regardless of what it
#' declares, because a clock band can only gate a parseable throughput
#' number; N regime-copies of a non-measurable bench would be N identical
#' records burning elevated lock time and risking a non-measurable row
#' leaking into a locked-perf bucket.
#'
#' NOTE this default differs deliberately from grid_measure.R:110, which
#' resolves an omitted `regimes` to the FULL clocks grid. That default is
#' right for a 7-cell sweep spec and catastrophic for a ~48-exe corpus:
#' it would silently multiply every un-swept exe across 6 clocks.
#'
#' @return character vector of regime labels (never empty).
effective_regimes <- function(cfg, warn = TRUE) {
  runnable   <- !identical(cfg$run, FALSE)
  measurable <- !identical(cfg$measurable, FALSE)
  declared   <- cfg$regimes %||% NULL

  if (!runnable || !measurable) {
    if (warn && length(declared))
      cat(sprintf("  WARN: %s declares regimes [%s] but is %s -- ignored, running once-native.\n",
                  cfg$id, paste(vapply(declared, regime_label_raw, character(1)), collapse = ", "),
                  if (!runnable) "run: false" else "measurable: false"))
    return("native")
  }
  if (!length(declared)) return("native")
  # unique(): a duplicated regime token would otherwise plan two cells
  # with the SAME (git_head, cell_id, regime) store key -- the collision
  # the three-tuple key exists to prevent (design Q3).
  unique(vapply(declared, regime_label_raw, character(1), USE.NAMES = FALSE))
}

#' Label a raw spec regime token without normalising through integer NA
#' (keeps "native" as "native" and 1605 as "1605").
regime_label_raw <- function(x) regime_label(normalise_clock(x))

#' Expand configs into the (config x regime) cell plan.
#'
#' The store key is `(git_head, cell_id, regime)` (design Q3), so a cell
#' is exactly one row of that key for this run. `band_mhz` resolves
#' per-config, falling back to the spec `defaults$band_mhz`.
#'
#' @param configs list of config records from merge_spec().
#' @param only_regime optional regime label; keep only cells in it. This
#'   is how a single R child measures one fixed regime -- pwsh unrolls the
#'   regimes, R never loops them (design "Orchestration flow").
#' @return list of cells: list(cfg, regime, clock_target_mhz, band_lo,
#'   band_hi, band_mhz).
plan_cells <- function(configs, only_regime = NULL, warn = TRUE) {
  cells <- list()
  for (cfg in configs) {
    for (rg in effective_regimes(cfg, warn = warn)) {
      if (!is.null(only_regime) && !identical(rg, only_regime)) next
      ct  <- normalise_clock(rg)
      # clock_band_for stops if a LOCKED cell has no usable band; native
      # passes band_mhz through untouched and gets NULL.
      bnd <- tryCatch(clock_band_for(ct, cfg$band_mhz),
                      error = function(e) stop(sprintf("%s: %s", cfg$id, conditionMessage(e)),
                                               call. = FALSE))
      cells[[length(cells) + 1L]] <- list(
        cfg = cfg, regime = rg, clock_target_mhz = ct,
        band_mhz = if (is.null(bnd)) NA_integer_ else as.integer(cfg$band_mhz),
        band_lo = if (is.null(bnd)) NA_integer_ else as.integer(bnd[1]),
        band_hi = if (is.null(bnd)) NA_integer_ else as.integer(bnd[2]))
    }
  }
  cells
}

#' Distinct regime labels in a cell plan, native first (the orchestrator
#' runs the lock-free group before touching -lgc), then ascending MHz.
#'
#' Sorted NUMERICALLY, not lexicographically: normalise_clock accepts
#' clocks down to 100 MHz, and a character sort would order "900" after
#' "1200". Phase 3's --plan-regimes handshake feeds this order to the
#' pwsh orchestrator as the clock-group sequence.
plan_regimes <- function(cells) {
  rg <- unique(vapply(cells, function(x) x$regime, character(1)))
  locked <- rg[rg != "native"]
  locked <- locked[order(vapply(locked, normalise_clock, integer(1), USE.NAMES = FALSE))]
  c(rg[rg == "native"], locked)
}

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
merge_spec <- function(corpus_src, spec_kernels, default_args,
                       spec_defaults = list()) {
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
      # PER-KERNEL override only. Do NOT fold spec_defaults$n_samples in
      # here: cfg$n_samples outranks --min-valid in measure_config, so a
      # folded default would make the CLI flag dead for every config and
      # silently raise the sample requirement corpus-wide. The spec
      # default is resolved at the opts level instead (see parse_args),
      # which keeps the precedence per-kernel > --min-valid > spec
      # default > built-in.
      n_samples   = k$n_samples %||% NULL,
      timeout     = k$timeout %||% NULL,
      # Regime fields (#152 Phase 2). `regimes` stays NULL when the spec
      # omits it -- effective_regimes() applies the [native] default, so
      # "omitted" and "explicitly native" stay distinguishable here.
      regimes     = k$regimes %||% NULL,
      warmup      = k$warmup   %||% spec_defaults$warmup   %||% NULL,
      band_mhz    = k$band_mhz %||% spec_defaults$band_mhz %||% NULL,
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

#' @param regime the clock regime this run measures ("native" or an MHz
#'   label). A run is regime-scoped (#152 Q3): pwsh starts one R child per
#'   clock group, so one results.json describes exactly one regime.
build_run_meta <- function(regime = "native") {
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
    regime         = regime,
    clock_lock     = regime,
    sm_arch        = "sm_86"
  )
}

#' Build a per-attempt JSONL row (also kept in results.json attempts[]).
#'
#' Store key is `(git_head, cell_id, regime)` (#152 Q3); `attempt` is a
#' row column, not part of the key. `cell` carries the regime context and
#' defaults to native so a caller that predates #152 still produces a
#' well-formed row.
attempt_row <- function(cfg, s, ok, reason, attempt, run_id, gh, cell = NULL) {
  post <- s$r$post
  cell <- cell %||% NATIVE_CELL
  list(
    run_id = run_id, ts_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC"),
    git_head = gh, cell_id = cfg$id, exe = cfg$exe, spec_source = cfg$spec_source,
    measurable = !identical(cfg$measurable, FALSE), verified = isTRUE(cfg$verified),
    args_str = paste(cfg$args, collapse = ","),
    regime = cell$regime,
    clock_target_mhz = as.integer(cell$clock_target_mhz),
    band_lo = as.integer(cell$band_lo), band_hi = as.integer(cell$band_hi),
    attempt = as.integer(attempt),
    ms = s$parsed$ms, throughput = s$parsed$throughput, unit = s$parsed$unit,
    clock_sm_mhz = as.integer((post$gpu$clock_sm %||% NA_integer_)),
    clock_mem_mhz = as.integer((post$gpu$clock_mem %||% NA_integer_)),
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

#' Measure one cell (config x regime): dispatch on run / measurable,
#' never throw.
#'
#' `cell` supplies the regime context (#152 Phase 2). Native cells carry
#' `band_lo`/`band_hi` NA, which resolves to `clock_band = NULL` -- the
#' exact pre-#152 behaviour. A locked cell passes the two-sided band that
#' `validate_sample` has always accepted; no new measurement path.
measure_config <- function(cfg, opts, jsonl_path, run_id, gh, cell = NULL) {
  cell <- cell %||% NATIVE_CELL
  clock_band <- cell_clock_band(cell)

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
                       1L, run_id, gh, cell)
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
    row <- attempt_row(cfg, s, ok, reason, attempt, run_id, gh, cell)
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
                                              s$r$pre, s$r$post, valid_when = vw,
                                              clock_band = clock_band),
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
  # min_valid starts NULL so an explicit `--min-valid 5` stays
  # distinguishable from "not passed". main() resolves it as
  # explicit CLI > spec defaults$n_samples > MIN_VALID_FALLBACK.
  out <- list(spec = DEFAULT_SPEC, out_dir = NULL, min_valid = NULL,
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
          "  --list  print the planned cell grid (config x regime) and exit (no GPU)\n",
          "\n",
          "Regimes (#152): a config's `regimes` in the spec lists the clocks it is\n",
          "worth measuring at; omitting it means [native]. This runner measures the\n",
          "NATIVE regime only -- locked regimes are planned and listed, but applying\n",
          "a host-side clock lock is the elevated pwsh orchestrator's job (Phase 3).\n",
          sep = "")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  out
}

#' Load the unified spec (#152 Q1): kernels + top-level `defaults` and
#' `clocks`.
#'
#' Both new top-level blocks are OPTIONAL, so a spec predating #152 still
#' loads and behaves exactly as before (no regimes -> every config plans
#' `[native]`). `clocks` is a fallback list a kernel opts into by writing
#' `regimes: <clocks>`; it is never applied implicitly -- see
#' effective_regimes() for why an implicit full-grid default is wrong for
#' a corpus this size.
#'
#' @return list(defaults, clocks, kernels).
load_spec <- function(spec_path) {
  empty <- list(defaults = list(), clocks = list(), kernels = list())
  if (!file.exists(spec_path)) {
    cat(sprintf("WARN: spec not found (%s); every bench runs default-args.\n", spec_path))
    return(empty)
  }
  spec <- yaml::read_yaml(spec_path)
  defaults <- spec$defaults %||% list()
  # Validate the regime-related defaults if present; a malformed band is a
  # spec bug worth failing on, not something to silently default away.
  if (!is.null(defaults$band_mhz)) {
    b <- suppressWarnings(as.integer(defaults$band_mhz))
    if (is.na(b) || b <= 0L)
      stop(sprintf("spec defaults$band_mhz must be a positive integer, got: %s",
                   as.character(defaults$band_mhz)))
    defaults$band_mhz <- b
  }
  clocks <- spec$clocks %||% list()
  for (cl in clocks) normalise_clock(cl)   # stops on a bad token
  list(defaults = defaults, clocks = clocks, kernels = spec$kernels %||% list())
}

#' Back-compat shim: the pre-#152 loader returned just the kernel list.
load_spec_kernels <- function(spec_path) load_spec(spec_path)$kernels

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))

  corpus  <- discover_corpus(REPO_ROOT)
  spec    <- load_spec(opts$spec)
  # Resolve the corpus-wide sample target: an explicit --min-valid wins,
  # then the spec's defaults$n_samples, then the built-in fallback. A
  # per-kernel `n_samples` still outranks all of these in measure_config.
  opts$min_valid <- as.integer(opts$min_valid %||% spec$defaults$n_samples %||%
                               MIN_VALID_FALLBACK)
  configs <- merge_spec(corpus, spec$kernels, opts$default_args, spec$defaults)
  if (!is.null(opts$only))
    configs <- Filter(function(c) identical(c$id, opts$only), configs)
  if (!length(configs)) { cat("No configs to run.\n"); quit(status = 1) }

  n_known   <- sum(vapply(configs, function(c) c$spec_source == "known", logical(1)))
  n_default <- length(configs) - n_known
  n_meas    <- sum(vapply(configs, function(c) isTRUE(c$measurable) && !identical(c$run, FALSE), logical(1)))

  # The FULL cell plan (every regime any config declares) is what --list
  # shows and what --plan-regimes will consume in Phase 3. The RUN plan is
  # a single regime: pwsh unrolls regimes one child per clock group, R
  # never loops them, and until Phase 3 lands --regime the only regime R
  # can honestly measure is the lock-free one.
  all_cells <- plan_cells(configs, warn = opts$list_only)
  regimes   <- plan_regimes(all_cells)

  cat(strrep("=", 72), "\n", sep = "")
  cat("  bench-all -- full-corpus run (skip nothing, record everything)\n")
  cat(sprintf("  corpus %d exes | configs %d (known %d, default %d) | measurable %d\n",
              length(corpus), length(configs), n_known, n_default, n_meas))
  cat(sprintf("  cells %d over %d regime(s): %s\n",
              length(all_cells), length(regimes), paste(regimes, collapse = ", ")))
  cat(sprintf("  min-valid %d | max-attempts %d | cooldown %.1fs\n",
              opts$min_valid, opts$max_attempts, opts$cooldown))
  cat(strrep("=", 72), "\n", sep = "")

  if (opts$list_only) {
    for (cell in all_cells) {
      c <- cell$cfg
      tag <- if (identical(c$run, FALSE)) "skip" else if (!isTRUE(c$measurable)) "nomeas" else "perf"
      band <- if (is.na(cell$band_lo)) "" else sprintf(" band=[%d,%d]", cell$band_lo, cell$band_hi)
      cat(sprintf("  [%-7s %-6s %-6s] %-30s  %s  args=[%s]%s\n",
                  c$spec_source, tag, cell$regime, c$id, c$exe,
                  paste(c$args, collapse = " "), band))
    }
    cat(sprintf("\n%d configs planned (%d known, %d default, %d measurable).\n",
                length(configs), n_known, n_default, n_meas))
    cat(sprintf("%d cells over regimes: %s\n", length(all_cells),
                paste(regimes, collapse = ", ")))
    for (rg in regimes)
      cat(sprintf("  regime %-8s : %d cell(s)\n", rg,
                  sum(vapply(all_cells, function(x) identical(x$regime, rg), logical(1)))))
    quit(status = 0)
  }

  # Phase 2 measures the native regime only. Locked regimes are planned
  # and reported above, but running one requires a host-side lock that
  # only the pwsh orchestrator can apply (#152 Phase 3) -- measuring a
  # locked cell without the lock would record a band-rejected or, worse,
  # a silently native number under a locked key.
  cells <- plan_cells(configs, only_regime = "native", warn = TRUE)
  n_deferred <- length(all_cells) - length(cells)
  if (n_deferred > 0L)
    cat(sprintf("  NOTE: %d locked-regime cell(s) deferred to the elevated orchestrator (#152 Phase 3).\n",
                n_deferred))

  # A config whose regime list omits `native` has NO cell in this run.
  # Say so by name. "skip nothing, record everything" is the banner
  # directly above; a config silently vanishing from results.json while
  # the header still counts it is the worst way to break that promise.
  measured_ids <- unique(vapply(cells, function(x) x$cfg$id, character(1)))
  unmeasured   <- setdiff(vapply(configs, function(c) c$id, character(1)), measured_ids)
  if (length(unmeasured))
    cat(sprintf(paste0("  NOTE: %d config(s) have NO native cell and are NOT measured by this run: %s\n",
                       "        (their regimes omit `native`; measure them via the elevated orchestrator)\n"),
                length(unmeasured), paste(unmeasured, collapse = ", ")))

  if (!length(cells)) {
    # Exiting 0 here would make `--only <locked-only-config>` a silent
    # no-op that writes an empty report and reads as success.
    cat("\nNothing to measure in the native regime.\n")
    if (!is.null(opts$only))
      cat(sprintf("  --only %s selected %d config(s), none of which has a native cell.\n",
                  opts$only, length(configs)))
    quit(status = 1)
  }

  run_id  <- format(Sys.time(), "%Y%m%dT%H%M%S")
  out_dir <- opts$out_dir %||% file.path(REPO_ROOT, "results", "bench_all", run_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  jsonl_path <- file.path(out_dir, "samples.jsonl")
  results_path <- file.path(out_dir, "results.json")
  summary_path <- file.path(out_dir, "summary.md")

  run_meta <- build_run_meta(regime = "native")
  gh <- run_meta$git_head
  .gs <- capture_gpu_state()
  if (!is.null(.gs)) {
    cat(sprintf("  GPU state: %s\n", summarise_meta(.gs, .gs)))
    cat(strrep("=", 72), "\n", sep = "")
  } else {
    cat("  WARNING: no GPU metadata (nvidia-smi absent) -- runs unguarded.\n")
  }

  summaries <- list()
  for (cell in cells) {
    cfg <- cell$cfg
    cat(sprintf("\n[%s] %s  (args=[%s], %s, regime=%s)\n", cfg$id, cfg$exe,
                paste(cfg$args, collapse = " "), cfg$spec_source, cell$regime))
    s <- tryCatch(measure_config(cfg, opts, jsonl_path, run_id, gh, cell),
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

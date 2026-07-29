# tests/bench_all/test_bench_all.R
#
# GPU-free unit tests for scripts/bench/bench_all.R (issue #124). Drives
# the pure functions (corpus discovery, spec merge, status classification,
# summary aggregation, markdown render) with the real repo + synthetic
# samples. The GPU glue (measure_config -> run_bench) needs a card and is
# exercised by an actual `make bench-all`, not here -- mirroring the
# project's GPU-free fixture-diff verification pattern.
#
# Run:
#   Rscript -e 'testthat::test_file("tests/bench_all/test_bench_all.R", stop_on_failure=TRUE)'
#   (or)  Rscript tests/bench_all/test_bench_all.R

library(testthat)

# Source bench_all.R for its functions. main() is guarded by
# `if (sys.nframe() == 0L) main()`, so sourcing runs no benchmarks.
.candidates <- c(
  "scripts/bench/bench_all.R",
  file.path(getwd(), "scripts", "bench", "bench_all.R"),
  "/mnt/d/dev/p/bare-metal/scripts/bench/bench_all.R")
.src <- NULL
for (.p in .candidates) if (file.exists(.p)) { .src <- .p; break }
if (is.null(.src)) stop("can't find bench_all.R")
suppressMessages(source(.src))

# ---- corpus discovery ------------------------------------------------
test_that("discover_corpus finds the bench corpus and excludes non-benches", {
  corpus <- discover_corpus(REPO_ROOT)
  # Every entry is a bench.cu or a kernels/**/bench_*.cu.
  expect_true(all(grepl("(^|/)bench\\.cu$|/bench_.*\\.cu$", corpus)))
  # Known members present.
  expect_true("kernels/gemm/hgemm/bench.cu" %in% corpus)
  expect_true("kernels/gemm/igemm/bench_sparse.cu" %in% corpus)
  expect_true("kernels/reference/cublas_hgemm/bench.cu" %in% corpus)
  # Pruned / non-corpus files excluded.
  expect_false(any(startsWith(corpus, "experiments/")))
  expect_false(any(startsWith(corpus, "tools/")))
  expect_false(any(grepl("^tests/", corpus)))          # tests/*.cu are not benches
  # A real corpus (the repo currently ships 48).
  expect_gte(length(corpus), 40L)
})

test_that("exe_for_src / auto_id", {
  expect_equal(exe_for_src("kernels/gemm/hgemm/bench.cu"), "kernels/gemm/hgemm/bench")
  expect_equal(auto_id("kernels/attention/cross_attention/bench_v2"),
               "attention_cross_attention_bench_v2")
})

# ---- spec merge ------------------------------------------------------
test_that("merge_spec marks known vs default and covers every exe once+", {
  corpus <- c("a/bench.cu", "b/bench_x.cu", "c/bench.cu")
  spec <- list(
    list(id = "ka", exe = "a/bench", args = list(2048, 2048), match = "foo"),
    list(id = "kb", exe = "b/bench_x", measurable = FALSE, note = "table"))
  cfgs <- merge_spec(corpus, spec, default_args = c("512"))
  src <- vapply(cfgs, function(c) c$spec_source, character(1))
  ids <- vapply(cfgs, function(c) c$id, character(1))
  # 2 known + 1 default (the uncovered c/bench).
  expect_equal(sum(src == "known"), 2L)
  expect_equal(sum(src == "default"), 1L)
  defcfg <- cfgs[[which(src == "default")]]
  expect_equal(defcfg$exe, "c/bench")
  expect_equal(defcfg$args, "512")               # default args applied
  expect_false(defcfg$measurable == FALSE)       # default is measurable
  # known fields carried through.
  ka <- cfgs[[which(ids == "ka")]]
  expect_equal(ka$args, c("2048", "2048"))
  expect_equal(ka$match, "foo")
  kb <- cfgs[[which(ids == "kb")]]
  expect_false(kb$measurable)
})

test_that("the shipped bench_all.yml covers the ENTIRE corpus (no default configs)", {
  corpus  <- discover_corpus(REPO_ROOT)
  spec_k  <- yaml::read_yaml(file.path(REPO_ROOT, "scripts", "bench", "bench_all.yml"))$kernels
  cfgs    <- merge_spec(corpus, spec_k, default_args = character(0))
  src     <- vapply(cfgs, function(c) c$spec_source, character(1))
  defaulted <- vapply(cfgs[src == "default"], function(c) c$exe, character(1))
  # A non-empty default set means a bench was added without a spec entry.
  expect_equal(length(defaulted), 0L,
               info = paste("un-specced exes:", paste(defaulted, collapse = ", ")))
  # Unique ids.
  ids <- vapply(cfgs, function(c) c$id, character(1))
  expect_equal(anyDuplicated(ids), 0L)
  # Every spec exe actually exists in the corpus.
  spec_exes <- vapply(spec_k, function(k) k$exe, character(1))
  expect_true(all(spec_exes %in% vapply(corpus, exe_for_src, character(1))),
              info = "a spec exe is not in the discovered corpus")
})

# ---- status classification ------------------------------------------
test_that("classify_status: ok / degraded / failed", {
  expect_equal(classify_status(TRUE, 5L), "ok")
  expect_equal(classify_status(FALSE, 2L), "degraded")
  expect_equal(classify_status(FALSE, 0L), "failed")
})

test_that("reason buckets + histogram + top_reject", {
  expect_equal(reason_bucket("crash(exit=1)"), "crash")
  expect_equal(reason_bucket("parse-fail"), "parse-fail")
  expect_equal(reason_bucket("unfair(SwPowerCap)"), "unfair")
  expect_equal(reason_bucket(NA_character_), "unknown")
  h <- reject_histogram(c("unfair(x)", "unfair(y)", "parse-fail"))
  expect_equal(h[["unfair"]], 2L)
  expect_equal(h[["parse-fail"]], 1L)
  expect_true(startsWith(top_reject(h), "unfair:2"))
})

test_that("pick_unit: spec unit wins, else parsed, else NA", {
  expect_equal(pick_unit("GB/s", c("GFLOPS")), "GB/s")        # spec authoritative
  expect_equal(pick_unit(NA_character_, c("TOPS")), "TOPS")   # fall to parsed
  expect_equal(pick_unit(NA_character_, character(0)), NA_character_)
})

# ---- summary aggregation --------------------------------------------
test_that("summarise_config: median/spread/status/verified, attempts kept", {
  cfg <- list(id = "k", exe = "e", src = "e.cu", args = c("2048"),
              spec_source = "known", verified = TRUE, unit = NA_character_,
              notes = "n")
  atts <- list(list(attempt = 1L), list(attempt = 2L), list(attempt = 3L))
  s <- summarise_config(cfg,
                        valid_tputs = c(100, 110, 120),
                        valid_mss   = c(1.0, 0.9, 0.8),
                        valid_units = c("GFLOPS", "GFLOPS", "GFLOPS"),
                        reject_reasons = c("unfair(x)"),
                        n_attempts = 4L, complete = TRUE, attempts = atts)
  expect_equal(s$status, "ok")
  expect_equal(s$n_valid, 3L)
  expect_equal(s$n_attempts, 4L)
  expect_equal(s$median_throughput, 110)
  expect_equal(s$tput_lo, 100); expect_equal(s$tput_hi, 120)
  expect_equal(s$unit, "GFLOPS")
  expect_true(s$verified)
  expect_length(s$attempts, 3L)         # every attempt retained verbatim
})

test_that("skeleton_summary shapes for not-built / skipped / non-measurable", {
  cfg <- list(id = "k", exe = "e", src = "e.cu", args = character(0),
              spec_source = "known", measurable = FALSE, verified = FALSE,
              unit = NA_character_, notes = "table")
  nb <- skeleton_summary(cfg, "not-built", 0L, 0L, "exe not built")
  expect_equal(nb$status, "not-built")
  expect_false(nb$measurable)
  nm <- skeleton_summary(cfg, "non-measurable", 0L, 1L, "ran; no number",
                         attempts = list(list(attempt = 1L)))
  expect_equal(nm$status, "non-measurable")
  expect_length(nm$attempts, 1L)
})

# ---- render: the advisor invariant ----------------------------------
test_that("render keeps measurable / non-measurable / default buckets separate", {
  meta <- list(ts_utc = "2026-06-04T00:00:00Z", git_head = "abc123",
               git_dirty = FALSE, host = "h", gpu_name = "g",
               driver_version = "1", sm_arch = "sm_86", nvcc = "CUDA 13",
               gpu_mode = "dgpu", clock_lock = "native")
  mk <- function(id, src, status, measurable, verified) list(
    id = id, exe = id, src = "x", args = character(0), spec_source = src,
    measurable = measurable, verified = verified, status = status,
    n_valid = 0L, n_attempts = 1L, median_throughput = NA_real_,
    median_ms = NA_real_, tput_lo = NA_real_, tput_hi = NA_real_,
    unit = NA_character_, reject_buckets = list(), top_reject = "parse-fail:1",
    notes = "")
  ss <- list(
    mk("perf_ok",   "known",   "ok",             TRUE,  TRUE),
    mk("nm_table",  "known",   "non-measurable", FALSE, FALSE),
    mk("def_fail",  "default", "failed",         TRUE,  FALSE))
  md <- render_summary_md(meta, ss)

  # Three labelled sections exist.
  expect_true(grepl("## Measurable corpus", md, fixed = TRUE))
  expect_true(grepl("## Non-measurable / skipped", md, fixed = TRUE))
  expect_true(grepl("## Discovered without a spec", md, fixed = TRUE))

  # The default-args FAILED config must NOT sit above the non-measurable
  # header (i.e. it is in the default bucket, not the measurable one) --
  # a default/parse-fail must never read as a real kernel failure.
  pos_meas    <- regexpr("## Measurable corpus", md, fixed = TRUE)
  pos_nonmeas <- regexpr("## Non-measurable / skipped", md, fixed = TRUE)
  pos_default <- regexpr("## Discovered without a spec", md, fixed = TRUE)
  pos_deffail <- regexpr("def_fail", md, fixed = TRUE)
  pos_nmtable <- regexpr("nm_table", md, fixed = TRUE)
  pos_perfok  <- regexpr("perf_ok", md, fixed = TRUE)
  expect_true(pos_perfok  > pos_meas    && pos_perfok  < pos_nonmeas) # measurable bucket
  expect_true(pos_nmtable > pos_nonmeas && pos_nmtable < pos_default) # non-measurable bucket
  expect_true(pos_deffail > pos_default)                              # default bucket
  # verified marker rendered.
  expect_true(grepl("verified", md, fixed = TRUE))
  expect_true(grepl("infer", md, fixed = TRUE))
})

# ---- regimes / cell plan (#152 Phase 2) ------------------------------
test_that("normalise_clock: native -> NA, MHz -> int, junk -> error", {
  expect_true(is.na(normalise_clock("native")))
  expect_true(is.na(normalise_clock("NATIVE")))
  expect_true(is.na(normalise_clock(NULL)))
  expect_true(is.na(normalise_clock(NA)))
  expect_equal(normalise_clock(1605), 1605L)
  expect_equal(normalise_clock("1605"), 1605L)
  expect_error(normalise_clock("nativ"))     # typo must NOT silently pass
  expect_error(normalise_clock(0))
  expect_error(normalise_clock(99))          # below the 100 MHz floor
  expect_error(normalise_clock(5001))        # above the 5000 MHz ceiling
})

test_that("regime_label / clock_band_for", {
  expect_equal(regime_label(NA_integer_), "native")
  expect_equal(regime_label(1605L), "1605")
  # Native gets NULL, not a wide band -- that is what preserves the
  # pre-#152 native behaviour exactly.
  expect_null(clock_band_for(NA_integer_, 30L))
  expect_equal(clock_band_for(1605L, 30L), c(1575, 1635))
})

test_that("effective_regimes: [native] default, and the taxonomy override", {
  # Omitted regimes -> [native], NOT the full clocks grid (the deliberate
  # divergence from grid_measure.R:110).
  expect_equal(effective_regimes(list(id = "a")), "native")
  expect_equal(effective_regimes(list(id = "a", regimes = list("native", 1605))),
               c("native", "1605"))
  # measurable:false / run:false run ONCE-NATIVE whatever they declare.
  expect_equal(effective_regimes(list(id = "nm", measurable = FALSE,
                                      regimes = list(1605, 1710)), warn = FALSE),
               "native")
  expect_equal(effective_regimes(list(id = "sk", run = FALSE,
                                      regimes = list(1605)), warn = FALSE),
               "native")
  # ...and say so rather than dropping them silently.
  expect_output(effective_regimes(list(id = "nm", measurable = FALSE,
                                       regimes = list(1605)), warn = TRUE),
                "ignored")
})

test_that("plan_cells expands config x regime and resolves bands", {
  cfgs <- list(
    list(id = "swept", measurable = TRUE, regimes = list("native", 1605), band_mhz = 30L),
    list(id = "plain", measurable = TRUE),
    list(id = "nomeas", measurable = FALSE, regimes = list(1605), band_mhz = 30L))
  cells <- plan_cells(cfgs, warn = FALSE)
  expect_equal(length(cells), 4L)             # 2 + 1 + 1 (once-native)
  rg <- vapply(cells, function(x) x$regime, character(1))
  expect_equal(rg, c("native", "1605", "native", "native"))
  # Native cells carry no band; the locked one does.
  expect_true(is.na(cells[[1]]$band_lo))
  expect_equal(cells[[2]]$band_lo, 1575L)
  expect_equal(cells[[2]]$band_hi, 1635L)
  expect_equal(cells[[2]]$clock_target_mhz, 1605L)
  expect_true(is.na(cells[[1]]$clock_target_mhz))
  # only_regime filters; this is how one R child measures one regime.
  native_only <- plan_cells(cfgs, only_regime = "native", warn = FALSE)
  expect_equal(length(native_only), 3L)
  expect_true(all(vapply(native_only, function(x) x$regime, character(1)) == "native"))
  locked_only <- plan_cells(cfgs, only_regime = "1605", warn = FALSE)
  expect_equal(length(locked_only), 1L)
  expect_equal(locked_only[[1]]$cfg$id, "swept")
  # plan_regimes lists native first (lock-free group runs before -lgc).
  expect_equal(plan_regimes(cells), c("native", "1605"))
})

test_that("a LOCKED cell with no usable band is a spec error, not a silent no-check", {
  # Degrading to "no band" would record an ungated number under a locked
  # key -- exactly the leak the two-sided band exists to prevent (Q4).
  locked <- list(id = "x", measurable = TRUE, regimes = list(1605))
  expect_error(plan_cells(list(locked), warn = FALSE), "band_mhz")
  expect_error(plan_cells(list(c(locked, list(band_mhz = 0))), warn = FALSE), "band_mhz")
  expect_error(plan_cells(list(c(locked, list(band_mhz = -30))), warn = FALSE), "band_mhz")
  expect_error(plan_cells(list(c(locked, list(band_mhz = "wide"))), warn = FALSE), "band_mhz")
  # The error names the offending config so a 50-entry spec is debuggable.
  expect_error(plan_cells(list(locked), warn = FALSE), "^x: ")
  # A native cell needs no band and must NOT error.
  expect_silent(plan_cells(list(list(id = "n", measurable = TRUE)), warn = FALSE))
  # ...and a valid band still resolves.
  ok <- plan_cells(list(c(locked, list(band_mhz = 30L))), warn = FALSE)
  expect_equal(ok[[1]]$band_lo, 1575L)
})

test_that("cell_clock_band is the single source of truth for the band", {
  # measure_config must not re-derive the band independently; a second
  # derivation is a second thing that can disagree with the recorded row.
  expect_null(cell_clock_band(NATIVE_CELL))
  expect_null(cell_clock_band(list(band_lo = NA_integer_, band_hi = 1635L)))
  expect_null(cell_clock_band(list(band_lo = NULL, band_hi = NULL)))
  expect_equal(cell_clock_band(list(band_lo = 1575L, band_hi = 1635L)), c(1575L, 1635L))
  # The planner and the helper agree on a real planned cell.
  cells <- plan_cells(list(list(id = "x", measurable = TRUE,
                                regimes = list(1605), band_mhz = 30L)), warn = FALSE)
  expect_equal(cell_clock_band(cells[[1]]), c(1575L, 1635L))
})

test_that("duplicate regime tokens collapse to one cell", {
  # Two cells with the same (git_head, cell_id, regime) would double a
  # cell's samples under one store key.
  cells <- plan_cells(list(list(id = "x", measurable = TRUE, band_mhz = 30L,
                                regimes = list(1605, "1605"))), warn = FALSE)
  expect_equal(length(cells), 1L)
})

test_that("plan_regimes orders locked regimes numerically, not lexically", {
  # 900 must precede 1200; a character sort puts it last. Phase 3's
  # orchestrator consumes this order as the -lgc group sequence.
  cells <- plan_cells(list(list(id = "x", measurable = TRUE, band_mhz = 30L,
                                regimes = list("native", 1710, 900, 1200))), warn = FALSE)
  expect_equal(plan_regimes(cells), c("native", "900", "1200", "1710"))
})

test_that("the shipped bench_all.yml plans a sane cell grid", {
  spec    <- load_spec(file.path(REPO_ROOT, "scripts", "bench", "bench_all.yml"))
  corpus  <- discover_corpus(REPO_ROOT)
  cfgs    <- merge_spec(corpus, spec$kernels, character(0), spec$defaults)
  cells   <- plan_cells(cfgs, warn = FALSE)
  # Every config appears at least once.
  expect_gte(length(cells), length(cfgs))
  # NB: do NOT loop `expect_silent(normalise_clock(cell$regime))` here.
  # cell$regime is by construction regime_label(normalise_clock(x)), and
  # plan_cells already calls normalise_clock on every label before
  # building the cell -- so an illegal token aborts plan_cells above and
  # the assertion can never fire. It only inflates the assertion count.
  # The real guard is the direct normalise_clock probe test.
  # Locked cells all carry a resolved band (defaults.band_mhz applies).
  locked <- Filter(function(x) !is.na(x$clock_target_mhz), cells)
  expect_gt(length(locked), 0L)
  expect_true(all(vapply(locked, function(x) !is.na(x$band_lo), logical(1))))
  # No non-measurable / skipped config ever leaves the native regime.
  for (cell in cells)
    if (identical(cell$cfg$measurable, FALSE) || identical(cell$cfg$run, FALSE))
      expect_equal(cell$regime, "native")
  # The seven grid_sweep.yml cells are reproduced with the same regimes.
  #
  # Matched by (exe, args), NOT by id: design Q3 asserts
  # `cell_id == bench_all id == grid id`, and that identity is currently
  # violated by exactly one cell (#160). Pin the known exception here so
  # a SECOND divergence cannot be introduced silently -- that is the whole
  # point of the store key.
  KNOWN_ID_MISMATCH <- c(conv2d_implicit_gemm_sd64 = "conv2d_implicit_gemm")  # #160
  grid <- yaml::read_yaml(file.path(REPO_ROOT, "scripts", "probe", "grid_sweep.yml"))
  key <- function(exe, args) paste(exe, paste(unlist(args %||% list()), collapse = ","), sep = "|")
  bkey <- vapply(cfgs, function(c) key(c$exe, c$args), character(1))
  for (gk in grid$kernels) {
    j <- which(bkey == key(gk$exe, gk$args))
    expect_length(j, 1L)                    # exactly one bench_all config per grid cell
    got <- cfgs[[j]]
    expected_id <- if (gk$id %in% names(KNOWN_ID_MISMATCH))
                     unname(KNOWN_ID_MISMATCH[gk$id]) else gk$id
    expect_equal(got$id, expected_id, info = paste("id drift for", gk$id))
    expect_equal(effective_regimes(got, warn = FALSE),
                 vapply(gk$regimes, regime_label_raw, character(1), USE.NAMES = FALSE),
                 info = gk$id)
  }
  # And the exception list must stay minimal: every OTHER grid id matches.
  gids <- vapply(grid$kernels, function(k) k$id, character(1))
  bids <- vapply(cfgs, function(c) c$id, character(1))
  expect_equal(setdiff(gids, bids), names(KNOWN_ID_MISMATCH))
})

test_that("--min-valid stays alive: spec defaults must not become a per-config override", {
  # Regression guard. Folding defaults$n_samples into every config's
  # n_samples silently kills --min-valid, because measure_config resolves
  # `cfg$n_samples %||% opts$min_valid` and cfg wins. Precedence must be
  # per-kernel n_samples > explicit --min-valid > defaults$n_samples > 5.
  spec <- load_spec(file.path(REPO_ROOT, "scripts", "bench", "bench_all.yml"))
  cfgs <- merge_spec(discover_corpus(REPO_ROOT), spec$kernels, character(0), spec$defaults)
  folded <- sum(vapply(cfgs, function(c) !is.null(c$n_samples), logical(1)))
  declared <- sum(vapply(spec$kernels, function(k) !is.null(k$n_samples), logical(1)))
  expect_equal(folded, declared,
               info = "spec defaults$n_samples leaked into per-config n_samples")
  # parse_args must leave min_valid NULL when the flag is absent, so main()
  # can tell "not passed" from "passed the same value as the default".
  expect_null(parse_args(character(0))$min_valid)
  expect_equal(parse_args(c("--min-valid", "3"))$min_valid, 3L)
  expect_equal(MIN_VALID_FALLBACK, 5L)
})

test_that("SKIP NOTHING survives regimes: every corpus exe is still covered natively", {
  # Regimes must not quietly drop an exe from the native pass. Today only
  # igemm_sparse_4096 omits `native`, and its exe is still reached by the
  # 2048 config -- but that is a property to VERIFY, not to assume, since
  # a future `regimes:` edit could orphan an exe with no other config.
  spec   <- load_spec(file.path(REPO_ROOT, "scripts", "bench", "bench_all.yml"))
  corpus <- discover_corpus(REPO_ROOT)
  cfgs   <- merge_spec(corpus, spec$kernels, character(0), spec$defaults)
  native <- plan_cells(cfgs, only_regime = "native", warn = FALSE)
  covered <- unique(vapply(native, function(x) x$cfg$exe, character(1)))
  uncovered <- setdiff(vapply(corpus, exe_for_src, character(1)), covered)
  expect_equal(length(uncovered), 0L,
               info = paste("exe(s) with no native cell:", paste(uncovered, collapse = ", ")))
})

test_that("attempt_row carries the regime columns and defaults to native", {
  cfg <- list(id = "k", exe = "e", spec_source = "known", args = c("1"),
              measurable = TRUE, verified = TRUE)
  s <- list(parsed = list(ms = 1, throughput = 100, unit = "GFLOPS"),
            r = list(rc = 0L, post = list(gpu = list(clock_sm = 1605, clock_mem = 7001,
                                                     power_w = 100, temp_c = 70,
                                                     throttle = "GpuIdle"),
                                          host = list(gpu_mode = "unknown"))))
  cell <- list(regime = "1605", clock_target_mhz = 1605L,
               band_lo = 1575L, band_hi = 1635L)
  row <- attempt_row(cfg, s, TRUE, NA_character_, 1L, "run1", "abc", cell)
  expect_equal(row$regime, "1605")
  expect_equal(row$clock_target_mhz, 1605L)
  expect_equal(row$band_lo, 1575L); expect_equal(row$band_hi, 1635L)
  expect_equal(row$clock_mem_mhz, 7001L)
  expect_true(row$measurable); expect_true(row$verified)
  # The taxonomy columns must TRACK the config, not be hardcoded TRUE:
  # a hardcoded TRUE would promote every non-measurable confirming run
  # into the measurable bucket of the rollup (the Q4 advisor invariant).
  nm_cfg <- modifyList(cfg, list(measurable = FALSE, verified = FALSE))
  nm_row <- attempt_row(nm_cfg, s, TRUE, NA_character_, 1L, "run1", "abc", cell)
  expect_false(nm_row$measurable); expect_false(nm_row$verified)
  # No cell -> native with NO band, so a pre-#152 caller still yields a
  # valid row and never claims a band it did not gate on.
  bare <- attempt_row(cfg, s, TRUE, NA_character_, 1L, "run1", "abc")
  expect_equal(bare$regime, "native")
  expect_true(is.na(bare$clock_target_mhz))
  expect_true(is.na(bare$band_lo)); expect_true(is.na(bare$band_hi))
})

test_that("load_spec reads defaults/clocks and rejects a bad band", {
  spec <- load_spec(file.path(REPO_ROOT, "scripts", "bench", "bench_all.yml"))
  expect_equal(spec$defaults$band_mhz, 30L)
  expect_equal(spec$defaults$n_samples, 7)
  expect_true(length(spec$clocks) > 0L)
  expect_true(length(spec$kernels) > 0L)
  # A spec with neither block still loads (pre-#152 compatibility).
  tf <- tempfile(fileext = ".yml")
  writeLines(c("kernels:", "  - id: a", "    exe: a/bench"), tf)
  old <- load_spec(tf)
  expect_equal(length(old$kernels), 1L)
  expect_equal(length(old$defaults), 0L)
  expect_equal(effective_regimes(merge_spec(character(0), old$kernels,
                                            character(0), old$defaults)[[1]]),
               "native")
  bad <- tempfile(fileext = ".yml")
  writeLines(c("defaults:", "  band_mhz: 0", "kernels: []"), bad)
  expect_error(load_spec(bad), "band_mhz")
})

# ---- collector (#152 Phase 2) ----------------------------------------
# Sourced into its OWN environment: bench_all_collect.R defines `%||%`,
# REPO_ROOT, parse_args and main too, and sourcing it at top level would
# silently shadow bench_all.R's versions for every test below this point.
# A missing file is a hard stop, not a skip -- this file IS the Phase 2
# deliverable, so "absent" must fail the suite, exactly as bench_all.R does.
.collect_candidates <- c(
  "scripts/bench/bench_all_collect.R",
  file.path(REPO_ROOT, "scripts", "bench", "bench_all_collect.R"))
.csrc <- NULL
for (.p in .collect_candidates) if (file.exists(.p)) { .csrc <- .p; break }
if (is.null(.csrc)) stop("can't find bench_all_collect.R")
.collect <- new.env(parent = globalenv())
suppressMessages(sys.source(.csrc, envir = .collect))
bind_rows_fill   <- .collect$bind_rows_fill
normalise_schema <- .collect$normalise_schema
collect_parse_args <- .collect$parse_args

test_that("sourcing the collector did not shadow bench_all.R's own symbols", {
  # parse_args must still be bench_all's (spec/out_dir/min_valid/...),
  # not the collector's (results_dir/jsonl/out/print).
  expect_true("min_valid" %in% names(parse_args(character(0))))
  expect_false("results_dir" %in% names(parse_args(character(0))))
})

test_that("bind_rows_fill tolerates schema drift across runs", {
  old_row <- list(cell_id = "k", throughput = 100, valid = TRUE)          # pre-#152
  new_row <- list(cell_id = "k", throughput = 110, valid = TRUE,
                  regime = "1605", clock_target_mhz = 1605L)              # post-#152
  dt <- bind_rows_fill(list(old_row, new_row))
  expect_equal(nrow(dt), 2L)
  expect_true("regime" %in% names(dt))
  expect_true(is.na(dt$regime[1]))            # filled, not dropped
  expect_equal(dt$regime[2], "1605")
  expect_equal(nrow(bind_rows_fill(list())), 0L)
})

test_that("normalise_schema backfills native ONLY when no clock target says otherwise", {
  dt <- normalise_schema(bind_rows_fill(list(
    list(cell_id = "k", throughput = 100, valid = TRUE))))
  # bench_all had no other mode before #152, so "native" is a fact here.
  expect_equal(dt$regime, "native")
  expect_true(all(c("clock_target_mhz", "band_lo", "band_hi",
                    "clock_mem_mhz", "measurable", "verified") %in% names(dt)))
  # A recorded regime is never overwritten.
  dt2 <- normalise_schema(bind_rows_fill(list(
    list(cell_id = "k", throughput = 100, valid = TRUE, regime = "1605"))))
  expect_equal(dt2$regime, "1605")
  # A foreign store (e.g. grid) has clock_target_mhz and NO regime key.
  # Backfilling those to "native" would average 1200/1605 MHz samples
  # into one bucket -- derive the label from the clock target instead.
  dt3 <- normalise_schema(bind_rows_fill(list(
    list(cell_id = "k", throughput = 50497, valid = TRUE, clock_target_mhz = 1605L),
    list(cell_id = "k", throughput = 38000, valid = TRUE, clock_target_mhz = 1200L),
    list(cell_id = "k", throughput = 27000, valid = TRUE, clock_target_mhz = NA_integer_))))
  expect_equal(dt3$regime, c("1605", "1200", "native"))
  expect_equal(nrow(normalise_schema(data.table::data.table())), 0L)
})

test_that("collector refuses to clobber the canonical rollup on a targeted collect", {
  # --jsonl is a hand-picked, possibly-foreign view; writing it to the
  # default path would silently replace the cross-run rollup.
  expect_error(collect_parse_args(c("--jsonl", "/tmp/x.jsonl")), "--out")
  expect_silent(collect_parse_args(c("--jsonl", "/tmp/x.jsonl", "--out", "/tmp/y.rds")))
  # A trailing valueless flag must stop, not silently produce NA.
  expect_error(collect_parse_args("--out"), "needs a value")
  expect_error(collect_parse_args("--jsonl"), "needs a value")
})

cat("bench_all.R unit tests defined.\n")

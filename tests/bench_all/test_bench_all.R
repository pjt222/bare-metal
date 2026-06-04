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

cat("bench_all.R unit tests defined.\n")

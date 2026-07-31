# tests/bench_regress/test_verdict.R
#
# GPU-free unit tests for the run verdict in scripts/bench/bench_regress.R
# (issue #176).
#
# The bug these pin: the verdict branched on `regressions > 0L` alone, so a
# run in which every config SKIPPED printed "PASSED -- all benchmarks within
# tolerance" and exited 0 having measured nothing. It did that on five
# consecutive real pushes at 7 of 7 skipped, and the pre-push hook rendered it
# as a green "All benchmarks within tolerance. Push allowed."
#
# summarise_verdict() is a pure function of four counters, so the whole
# decision is testable here with no GPU, no baselines file and no bench
# executables. What is NOT covered here is the wiring -- whether main() feeds
# it the right counters. That needs a real run; the two recorded on PR for
# #176 are an all-skip (Total 2, Measured 0 -> INCONCLUSIVE, exit 2) and a
# live one (Total 7, Measured 3 -> PASSED, exit 0).
#
# Run:
#   Rscript tests/bench_regress/test_verdict.R

library(testthat)

# Source bench_regress.R for its functions. main() is guarded by
# `if (sys.nframe() == 0L) main()`, so sourcing runs no benchmarks.
#
# Two relative candidates only, both resolved from the repo root, which is
# where scripts/audit/run_r_tests.R runs its children. The sibling suites end
# this list with a hardcoded /mnt/d/dev/p/bare-metal/... path that makes them
# pass on one machine and fail everywhere else (#173) -- not repeated here.
.candidates <- c(
  "scripts/bench/bench_regress.R",
  file.path(getwd(), "scripts", "bench", "bench_regress.R"))
.src <- NULL
for (.p in .candidates) if (file.exists(.p)) { .src <- .p; break }
if (is.null(.src)) {
  stop("can't find scripts/bench/bench_regress.R -- run this from the repo root")
}
suppressMessages(source(.src))

# ---- the reported bug --------------------------------------------------

test_that("an all-skipped run is INCONCLUSIVE, not PASSED", {
  # The #176 observation, verbatim: 7 configs, every one skipped.
  v <- summarise_verdict(total = 7L, measured = 0L, regressions = 0L,
                         skipped = 7L)
  expect_equal(v$status, "INCONCLUSIVE")
  expect_equal(v$exit, 2L)
})

test_that("an all-skipped run never claims benchmarks are within tolerance", {
  # The precise sentence the operator read as a pass. It must not survive
  # anywhere in the message, whatever else the wording becomes.
  v <- summarise_verdict(7L, 0L, 0L, 7L)
  expect_false(grepl("within tolerance", v$msg, fixed = TRUE))
  expect_false(grepl("PASSED", v$msg, fixed = TRUE))
})

test_that("an all-skipped run says what it did not do", {
  v <- summarise_verdict(7L, 0L, 0L, 7L)
  expect_match(v$msg, "0 of 7", fixed = TRUE)
  expect_match(v$msg, "nothing was verified", fixed = TRUE)
})

test_that("an all-skipped run does not exit 0", {
  # The whole failure was a zero exit status: the hook branches on it.
  expect_gt(summarise_verdict(7L, 0L, 0L, 7L)$exit, 0L)
})

# ---- the empty corpus --------------------------------------------------

test_that("zero configs is INCONCLUSIVE, not a vacuous pass", {
  # No baselines matched, or nothing was built: 0/0 is not a clean run.
  v <- summarise_verdict(0L, 0L, 0L, 0L)
  expect_equal(v$status, "INCONCLUSIVE")
  expect_equal(v$exit, 2L)
})

# ---- the healthy run ---------------------------------------------------

test_that("a fully measured clean run is PASSED and exits 0", {
  v <- summarise_verdict(7L, 7L, 0L, 0L)
  expect_equal(v$status, "PASSED")
  expect_equal(v$exit, 0L)
  expect_match(v$msg, "within tolerance", fixed = TRUE)
})

test_that("a partly measured clean run states the fraction it measured", {
  # The routine case on this laptop: some configs behind a clock lock the
  # hook never applies (#156), some lost to throttle. It is a pass, but a
  # pass over 3 configs must not read like a pass over 7.
  v <- summarise_verdict(7L, 3L, 0L, 4L)
  expect_equal(v$status, "PASSED")
  expect_equal(v$exit, 0L)
  expect_match(v$msg, "3 of 7", fixed = TRUE)
  expect_match(v$msg, "4 skipped", fixed = TRUE)
})

test_that("one measured config is enough to be conclusive", {
  # The floor is one: measuring something is the difference the verdict
  # turns on, not measuring most things.
  v <- summarise_verdict(7L, 1L, 0L, 6L)
  expect_equal(v$status, "PASSED")
  expect_equal(v$exit, 0L)
})

test_that("improvements do not change the verdict", {
  # `improvements` is reported in the counts line but is deliberately not an
  # argument here: a faster kernel is still a pass.
  expect_equal(summarise_verdict(2L, 2L, 0L, 0L)$status, "PASSED")
})

# ---- regressions outrank everything ------------------------------------

test_that("a measured regression is FAILED and exits 1", {
  v <- summarise_verdict(7L, 7L, 2L, 0L)
  expect_equal(v$status, "FAILED")
  expect_equal(v$exit, 1L)
  expect_match(v$msg, "2 regression", fixed = TRUE)
})

test_that("a regression outranks a mostly-skipped run", {
  # 1 measured, 6 skipped, and the one measured config regressed. That is
  # real information: FAILED, not INCONCLUSIVE.
  v <- summarise_verdict(7L, 1L, 1L, 6L)
  expect_equal(v$status, "FAILED")
  expect_equal(v$exit, 1L)
})

test_that("FAILED still reports how much was measured", {
  # A regression found in 1 of 7 configs is weaker evidence than one found
  # in 7 of 7, and the message has to let the reader tell them apart.
  expect_match(summarise_verdict(7L, 1L, 1L, 6L)$msg, "1 of 7", fixed = TRUE)
})

# ---- the exit codes are the interface ----------------------------------

test_that("the three outcomes have three distinct exit codes", {
  # .githooks/pre-push and scripts/probe/run_locked_eval.ps1 both branch on
  # these numbers. Collapsing any two would silently change hook policy.
  codes <- c(
    passed       = summarise_verdict(7L, 7L, 0L, 0L)$exit,
    failed       = summarise_verdict(7L, 7L, 1L, 0L)$exit,
    inconclusive = summarise_verdict(7L, 0L, 0L, 7L)$exit)
  expect_equal(unname(codes), c(0L, 1L, 2L))
  expect_equal(length(unique(codes)), 3L)
})

test_that("every verdict carries a status, an exit code and a message", {
  for (v in list(summarise_verdict(7L, 7L, 0L, 0L),
                 summarise_verdict(7L, 7L, 1L, 0L),
                 summarise_verdict(7L, 0L, 0L, 7L))) {
    expect_true(is.character(v$status) && nzchar(v$status))
    expect_true(is.numeric(v$exit))
    expect_true(is.character(v$msg) && nzchar(v$msg))
    # The RESULT line is printed as "RESULT: <msg>", so the message has to
    # start with the status word or the two disagree on screen.
    expect_true(startsWith(v$msg, v$status))
  }
})

# ---- end-to-end: the wiring, not just the decision ---------------------
#
# Everything above tests summarise_verdict() in isolation. That is not enough on
# its own: the verdict function could be perfect and never consulted, or fed the
# wrong counters, and every group above would still pass. #176 was a wiring bug
# -- the decision code was three lines and they were all reachable.
#
# So these run the real script as a subprocess against a throwaway repo: a
# renv.lock (the marker REPO_ROOT walks up to), a data/baselines.json, and a copy
# of bench_regress.R. No GPU, no nvcc, no built corpus -- the "benchmark" is a
# shell script printing one bench-shaped line, and the fixture switches off the
# fairness check (default_valid_when: require_no_throttle false) so the result
# does not depend on what this machine's GPU happens to be doing at the time.
#
# The child runs with the working directory left at the repo root so that the
# repo's .Rprofile activates renv and cuasmR is loadable; REPO_ROOT comes from
# the script's own path, which is inside the fixture.

.fixture_root <- function(throughput = NULL, baseline_gflops = 1000) {
  root <- file.path(tempfile("bench_regress_fixture"))
  dir.create(file.path(root, "scripts", "bench"), recursive = TRUE)
  dir.create(file.path(root, "data"), recursive = TRUE)
  # REPO_ROOT walks up until it finds .git or renv.lock.
  file.create(file.path(root, "renv.lock"))
  file.copy(.src, file.path(root, "scripts", "bench", "bench_regress.R"))

  # throughput = NULL means "leave the executable missing", which is the
  # unbuilt-corpus case: the configs must still be counted.
  exe <- file.path(root, "fake_bench")
  if (!is.null(throughput)) {
    writeLines(c("#!/bin/sh",
                 sprintf('echo "  fixture 1.000 ms  %s GFLOPS"', throughput)),
               exe)
    Sys.chmod(exe, "0755")
  }

  baselines <- list(
    recorded_date = "fixture",
    platform = "fixture",
    default_valid_when = list(require_no_throttle = FALSE),
    kernels = list(`kernels/fixture/fake.cu` = list(
      exe = exe,
      `1_2` = list(ms = 1.0, gflops = baseline_gflops))))
  writeLines(jsonlite::toJSON(baselines, auto_unbox = TRUE, pretty = TRUE),
             file.path(root, "data", "baselines.json"))
  root
}

.run_gate <- function(root) {
  rscript <- file.path(R.home("bin"), "Rscript")
  out <- suppressWarnings(system2(
    rscript, file.path(root, "scripts", "bench", "bench_regress.R"),
    stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  list(status = if (is.null(status)) 0L else as.integer(status),
       text = paste(out, collapse = "\n"))
}

test_that("end to end: a run that skips everything exits 2 and says INCONCLUSIVE", {
  # This is #176 itself. Against the code as it was, this exits 0 and prints
  # "PASSED -- all benchmarks within tolerance".
  r <- .run_gate(.fixture_root(throughput = NULL))
  expect_equal(r$status, 2L)
  expect_match(r$text, "INCONCLUSIVE", fixed = TRUE)
  expect_false(grepl("within tolerance", r$text, fixed = TRUE))
})

test_that("end to end: configs of an unbuilt kernel stay in the denominator", {
  # The emptier version of the same bug: the missing-executable branch used to
  # skip the whole kernel before any counter moved, reporting `Total: 0`.
  r <- .run_gate(.fixture_root(throughput = NULL))
  expect_match(r$text, "Total: 1", fixed = TRUE)
  expect_match(r$text, "Measured: 0", fixed = TRUE)
  expect_match(r$text, "0 of 1", fixed = TRUE)
})

test_that("end to end: a config that measures cleanly exits 0 and is counted", {
  r <- .run_gate(.fixture_root(throughput = 1000))
  expect_equal(r$status, 0L)
  expect_match(r$text, "Measured: 1", fixed = TRUE)
  expect_match(r$text, "1 of 1 config(s) measured", fixed = TRUE)
})

test_that("end to end: a measured regression exits 1", {
  # Half of baseline, far outside the 10% default tolerance.
  r <- .run_gate(.fixture_root(throughput = 500))
  expect_equal(r$status, 1L)
  expect_match(r$text, "FAILED", fixed = TRUE)
  expect_match(r$text, "Measured: 1", fixed = TRUE)
})

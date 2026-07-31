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

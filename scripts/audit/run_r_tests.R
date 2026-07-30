#!/usr/bin/env Rscript
# run_r_tests.R -- the GPU-free R test gate (#163).
#
# Single entry point for every GPU-free R test suite in the repo. Called by
# `make test-r` and by .githooks/pre-push, so hook and CI run one thing rather
# than duplicating an invocation string.
#
# Until #163 these suites were invoked by nothing -- not the hook, not the
# Makefile, not CI. They only ran when someone remembered. The assertions they
# hold are regression guards written *because* the corresponding bug already
# happened once (the --min-valid precedence fold, KNOWN_ID_MISMATCH pinning the
# one tolerated grid/bench_all cell-id divergence, the "SKIP NOTHING survives
# regimes" check). Unwired, they guard nothing.
#
# WHY A CHILD PROCESS PER SUITE, RUN SERIALLY
#
# Per suite, not one shared process: each test file calls source() at top level
# to pull in the script under test, and source() defaults to local = FALSE --
# it writes into globalenv() even when testthat evaluates the file in its own
# environment. scripts/bench/bench_all.R and scripts/bench/bench_regress.R both
# define `main` and `parse_args`, so two suites sharing one process would see
# whichever sourced last. Today the per-file ordering happens to save it; that
# is a landmine, not a design. Separate processes also mean one suite crashing
# the interpreter cannot take the rest of the gate down with it.
#
# Serially, not in parallel: this repo lives on a 9p /mnt/d mount. Concurrent R
# processes hammering it is what turned a ~75 s suite into a 1 h one on
# 2026-07-29 (see #162 and the AGENTS.md startup-cost note). The ~5 s startup
# paid per child is the cheaper half of that trade.
#
# The child is a plain `Rscript`, deliberately. The
# `--no-init-file` + R_LIBS_USER bypass would shave the startup, but a bypass
# that loses cuasmR makes the gate pass *vacuously* -- worse than a slow gate.
# See run_grid_sweep.ps1:402-409 for the one place that bypass is load-bearing.
#
# WHY `Rscript <file>` AND NOT `testthat::test_file(<file>)`
#
# test_file() changes the working directory to the test file's own directory for
# the duration of the run. Measured 2026-07-30 with a probe test: under
# test_file() getwd() is <repo>/tests/<subdir>; under `Rscript <file>` it stays
# at the repo root.
#
# That matters because each repo-level suite resolves the script under test
# through a candidate list whose last entry is the hardcoded absolute path
# /mnt/d/dev/p/bare-metal/... (test_bench_all.R:21, test_meta.R:17). Under
# test_file() the two relative candidates miss and the hardcoded one hits, so
# the suite passes on this machine and dies with "can't find bench_all.R" on any
# other -- a CI-only failure that looks green in every local run. `Rscript <file>`
# from the repo root makes the first, relative candidate hit, which is the one
# that is actually portable.
#
# A useful side effect: under `Rscript <file>` a failing test_that aborts the
# script, so a suite's trailing top-level cat() is never reached. Under
# test_file() it prints even when every group in the file errored -- the retired
# test_parser.R printed "All bench_regress parser tests passed." while all 14 of
# its groups were erroring, which is how it stayed dead for 58 days.
# tests/bench_regress/test_meta.R:192 still ends in such a line
# ("All bench_meta tests passed."). Either way the verdict here comes from the
# child's exit status, never from what the child wrote.
#
# testthat DEFAULTS WORTH KNOWING
#
#   * stop_on_failure defaults to FALSE for test_file()/test_dir(), which exit 0
#     on a failing suite. Any invocation form used here must exit non-zero.
#   * max_fails defaults to 10, so a suite failing harder than that reports a
#     truncated tally (tests/bench_regress/test_parser.R has 14 failing groups;
#     the default showed 10). Irrelevant to the `Rscript <file>` form, which stops
#     at the first failure by design, but set explicitly for the package suite.
#
# WHY --expect EXISTS
#
# "All discovered suites passed" is a ratio against whatever was discovered, so
# it is always 100%. Delete every suite and this script still exits 0, reporting
# 0/0. A gate whose denominator is supplied by the thing it is checking cannot
# detect its own erosion -- and erosion by silence is precisely the failure #163
# was filed about. `--expect N` supplies the denominator from outside. CI passes
# it directly; the hook passes it transitively through `make test-r`, so the
# value itself lives in exactly two places -- Makefile's R_SUITES and tests.yml.
# Changing the suite count is then a deliberate edit rather than something that
# happens to you.
#
# Usage:
#   Rscript scripts/audit/run_r_tests.R             # run every suite, report, exit 0/1
#   Rscript scripts/audit/run_r_tests.R --expect 3  # ...and fail unless exactly 3 were found
#   Rscript scripts/audit/run_r_tests.R --list      # print the discovered suites, run nothing
#   Rscript scripts/audit/run_r_tests.R --quiet     # suppress this script's own
#                                                   # banners and summary table.
#                                                   # testthat's report still
#                                                   # prints: children stream.
#
# Exit codes:
#   0  every suite passed
#   1  a suite failed or errored, none were discovered, or the discovered count
#      did not match --expect

args  <- commandArgs(trailingOnly = TRUE)
quiet <- "--quiet" %in% args
list_only <- "--list" %in% args

expect_n <- NA_integer_
if ("--expect" %in% args) {
  i <- match("--expect", args)
  if (i == length(args)) {
    cat("run_r_tests.R: --expect needs a value\n")
    quit(status = 1)
  }
  expect_n <- suppressWarnings(as.integer(args[i + 1L]))
  if (is.na(expect_n) || expect_n < 0L) {
    cat("run_r_tests.R: --expect needs a non-negative integer, got: ",
        args[i + 1L], "\n", sep = "")
    quit(status = 1)
  }
  args <- args[-c(i, i + 1L)]
}

unknown <- setdiff(args, c("--quiet", "--list"))
if (length(unknown)) {
  cat("run_r_tests.R: unknown argument(s): ", paste(unknown, collapse = " "), "\n", sep = "")
  cat("Usage: Rscript scripts/audit/run_r_tests.R [--list] [--quiet] [--expect N]\n")
  quit(status = 1)
}

# Repo root: this script lives at scripts/audit/, so two levels up. Fall back to
# the working directory when the path cannot be recovered (e.g. `Rscript -e
# source(...)`), which is how the hook and Makefile both invoke it anyway.
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
repo_root <- if (!is.na(this_file) && nzchar(this_file)) {
  normalizePath(file.path(dirname(this_file), "..", ".."), mustWork = FALSE)
} else {
  normalizePath(getwd(), mustWork = FALSE)
}

# ---- discovery -------------------------------------------------------------
#
# Globbed, not a hardcoded manifest. A hardcoded list reproduces the exact bug
# #163 exists to fix: a suite added later is silently ungated. Anything matching
# tests/**/test_*.R or tests/**/test-*.R is picked up the day it lands.
#
# BOTH separators, deliberately. The two files under tests/ happen to use
# `test_`, but `test-` is what testthat's own convention produces --
# usethis::use_test() emits it, and R/cuasmR/tests/testthat/ is already all
# `test-`. An underscore-only glob would silently skip the next suite someone
# creates the ordinary way, which is #163 verbatim, re-armed.
suites <- list.files(
  file.path(repo_root, "tests"),
  pattern    = "^test[-_].*\\.R$",
  recursive  = TRUE,
  full.names = TRUE)
suites <- sort(suites)

# The cuasmR package carries its own testthat suite. It is GPU-free and the
# package is already on the library path (scripts/bench/bench_regress.R uses
# cuasmR::), so there is no extra install cost to running it here.
cuasmr_tests <- file.path(repo_root, "R", "cuasmR", "tests", "testthat")
has_cuasmr <- dir.exists(cuasmr_tests) &&
  length(list.files(cuasmr_tests, pattern = "^test-.*\\.R$")) > 0L

# fixed=TRUE, not a regex: a checkout path containing "(", "+" or "[" would
# otherwise throw or mis-strip.
rel <- function(p) {
  prefix <- paste0(repo_root, "/")
  ifelse(startsWith(p, prefix), substring(p, nchar(prefix) + 1L), p)
}

if (length(suites) == 0L && !has_cuasmr) {
  cat("run_r_tests.R: no GPU-free R suites discovered under ", rel(file.path(repo_root, "tests")), "\n", sep = "")
  cat("This is a discovery failure, not an empty-but-healthy repo -- failing loudly\n")
  cat("rather than reporting a vacuous pass.\n")
  quit(status = 1)
}

n_total <- length(suites) + as.integer(has_cuasmr)

# `==` not identical(): identical() is type-strict, so if n_total ever became a
# double the gate would fail always, and the fix someone reaches for first is
# deleting --expect.
expect_ok <- is.na(expect_n) || n_total == expect_n

if (list_only) {
  cat("GPU-free R suites (", n_total, "):\n", sep = "")
  for (s in suites) cat("  ", rel(s), "\n", sep = "")
  if (has_cuasmr) cat("  ", rel(cuasmr_tests), " (cuasmR package suite)\n", sep = "")
  if (!expect_ok) {
    cat("\nEXPECTED ", expect_n, " suite(s), DISCOVERED ", n_total, ".\n", sep = "")
    quit(status = 1)
  }
  quit(status = 0)
}

if (!expect_ok) {
  cat("\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat("  GPU-free R suites: SUITE COUNT CHANGED\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat("  expected ", expect_n, ", discovered ", n_total, "\n\n", sep = "")
  for (s in suites) cat("  found  ", rel(s), "\n", sep = "")
  if (has_cuasmr) cat("  found  ", rel(cuasmr_tests), " (cuasmR package suite)\n", sep = "")
  cat("\n")
  cat("A suite was added, removed, or renamed out of the discovery pattern\n")
  cat("(^test[-_].*\\.R$ under tests/). If that was intentional, update the\n")
  cat("expected count in BOTH places that supply it:\n")
  cat("  Makefile                     R_SUITES ?= <n>\n")
  cat("  .github/workflows/tests.yml  --expect <n>\n")
  cat("If it was not, a suite just stopped being gated -- which is the exact\n")
  cat("failure #163 was filed about.\n")
  quit(status = 1)
}

if (!requireNamespace("testthat", quietly = TRUE)) {
  cat("run_r_tests.R: {testthat} is not installed -- cannot run the GPU-free suites.\n")
  cat("Install it with: Rscript -e 'renv::restore()'  (or make setup)\n")
  quit(status = 1)
}

# ---- run -------------------------------------------------------------------

rscript <- file.path(R.home("bin"), "Rscript")

# Count skipped tests in a captured child log.
#
# Two renderings, because the two invocation forms use different reporters:
#   * `Rscript <file>` (StopReporter) prints a "-- Skip: <name> --" block per
#     skip, with the reason on the next line.
#   * test_local() (ProgressReporter) prints one "[ FAIL n | WARN n | SKIP n |
#     PASS n ]" summary line.
# Prefer the summary line where present; fall back to counting blocks.
count_skips <- function(lines) {
  # Strip ANSI first. Merging stderr into a pipe means the child's stdout is not
  # a TTY, but cli special-cases GitHub Actions and colourises anyway -- and an
  # escape landing between "SKIP" and its digits would make the regex miss and
  # CI silently report 0 skips.
  lines <- gsub("\033\\[[0-9;]*m", "", lines, useBytes = TRUE)

  # Anchored on the whole summary shape rather than a bare "SKIP n", so a test
  # NAME containing the word cannot be mistaken for a tally. testthat prints one
  # such line per file plus a cumulative total last, hence the final match.
  summary_line <- grep("\\[\\s*FAIL.*SKIP\\s+[0-9]+", lines, value = TRUE)
  if (length(summary_line)) {
    n <- sub(".*SKIP\\s+([0-9]+).*", "\\1", summary_line[length(summary_line)])
    return(suppressWarnings(as.integer(n)))
  }
  # StopReporter (the `Rscript <file>` form) has no summary line; it prints a
  # "-- Skip: <name> --" block with the reason beneath. Verified 2026-07-30:
  #   Rscript -e 'library(testthat); test_that("t", skip("why"))'
  # emits "== Skip: t ===..." / "Reason: why", so this matches.
  sum(grepl("Skip:", lines, fixed = TRUE, useBytes = TRUE))
}

run_child <- function(cmd_args, label) {
  if (!quiet) {
    cat("\n", strrep("-", 72), "\n", sep = "")
    cat("  ", label, "\n", sep = "")
    cat(strrep("-", 72), "\n", sep = "")
  }
  started <- Sys.time()
  logfile <- tempfile("r_suite_", fileext = ".log")
  on.exit(unlink(logfile), add = TRUE)

  # The output is both streamed and captured. Streamed because a suite can run
  # for minutes on the 9p mount and a silent hook reads as a hung one; captured
  # because the verdict needs a skip count, and a suite whose tests all skipped
  # must not be reportable as a clean pass (that is how the roundtrip coverage
  # would vanish the moment nvdisasm left PATH).
  #
  # `tee` swallows the child's exit status, so pipefail is required. testthat
  # writes its report to stderr, hence 2>&1 -- never redirect it away.
  cmdline <- paste(c(shQuote(rscript), cmd_args), collapse = " ")
  code <- system2("/bin/bash", c("-c", shQuote(sprintf(
    "set -o pipefail; %s 2>&1 | tee %s", cmdline, shQuote(logfile)))))

  lines <- if (file.exists(logfile)) readLines(logfile, warn = FALSE) else character(0)
  list(code  = as.integer(code),
       skips = count_skips(lines),
       secs  = as.numeric(difftime(Sys.time(), started, units = "secs")))
}

results <- list()

# ---- plumbing canary -------------------------------------------------------
#
# The entire verdict flows through one shell option. run_child() relies on
# `set -o pipefail` to stop `tee` from swallowing the child's exit status; that
# is a bashism, so changing /bin/bash to /bin/sh on a dash system would make
# every suite report a pass with nothing else noticing. It is the same class of
# silent failure as reading $PIPESTATUS in zsh, where it is empty and a failing
# command reads as passing.
#
# So: run one child that does nothing but exit 3, and require that it comes back
# non-zero. --expect guards the denominator, the skip count guards coverage, and
# this guards the verdict itself. Costs one R startup.
canary <- run_child(c("-e", shQuote("quit(status = 3)")), "plumbing canary")
if (canary$code == 0L) {
  cat("\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat("  EXIT-STATUS PLUMBING IS BROKEN\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat("  A child that exited 3 was read as success, so every suite below would\n")
  cat("  report a vacuous pass. Check the `set -o pipefail` in run_child() and\n")
  cat("  that /bin/bash is really bash.\n")
  quit(status = 1)
}

# Children run from the repo root so each suite's *relative* source candidate
# resolves. See the test_file() note in the header -- this is load-bearing.
#
# Not paired with an on.exit() restore: at top level on.exit() registers nothing,
# and this script always ends in quit(), which does not run exit handlers anyway.
# The cwd change dies with the process, which is the whole scope that matters.
setwd(repo_root)

for (s in suites) {
  r <- run_child(c(shQuote(rel(s))), rel(s))
  results[[rel(s)]] <- r
}

if (has_cuasmr) {
  # test_local() loads the package from SOURCE via pkgload, while the repo-level
  # suites load the INSTALLED cuasmR (scripts/bench/bench_meta.R does
  # library(cuasmR)). That divergence is deliberate: it is what catches an edit
  # to R/cuasmR/ that was never reinstalled. CI must `R CMD INSTALL R/cuasmR`
  # from the working tree so both halves see the same code.
  expr <- sprintf(
    'testthat::set_max_fails(Inf); testthat::test_local("%s", stop_on_failure = TRUE)',
    rel(dirname(dirname(cuasmr_tests))))
  r <- run_child(c("-e", shQuote(expr)),
                 paste0(rel(cuasmr_tests), " (cuasmR package suite)"))
  results[[rel(cuasmr_tests)]] <- r
}

# ---- report ----------------------------------------------------------------

failed <- names(results)[vapply(results, function(r) r$code != 0L, logical(1))]
total_secs  <- sum(vapply(results, function(r) r$secs, numeric(1)))
total_skips <- sum(vapply(results, function(r) r$skips, integer(1)), na.rm = TRUE)

# Skips are reported, always, and per suite. A skip is coverage that did not
# happen: three of the cuasmR roundtrip tests skip without a built cubin or
# without nvdisasm on PATH, and one test_meta.R test skips without nvidia-smi.
# Printed only as a total, "3/3 PASSED (4 skipped)" would read as a clean run to
# anyone not counting -- and the byte-identical roundtrip check is exactly the
# coverage you would least want to lose without noticing.
suite_line <- function(nm, mark) {
  sk <- results[[nm]]$skips
  sprintf("  %-4s %-48s %5.0f s%s\n", mark, nm, results[[nm]]$secs,
          if (!is.na(sk) && sk > 0L) sprintf("  (%d skipped)", sk) else "")
}

if (length(failed) == 0L) {
  if (!quiet) {
    cat("\n")
    cat(strrep("=", 72), "\n", sep = "")
    cat("  GPU-free R suites: ", n_total, "/", n_total, " PASSED",
        sprintf("  (%.0f s", total_secs),
        if (total_skips > 0L) sprintf(", %d skipped)", total_skips) else ")",
        "\n", sep = "")
    cat(strrep("=", 72), "\n", sep = "")
    for (nm in names(results)) cat(suite_line(nm, "ok"))
    if (total_skips > 0L) {
      cat("\n")
      cat("  ", total_skips, " test(s) skipped -- that coverage did NOT run.\n", sep = "")
      cat("  Usually: no built cubin (make cubins), nvdisasm off PATH, or no\n")
      cat("  nvidia-smi. Expected on a CI runner; on a dev box it means the\n")
      cat("  roundtrip and live-capture checks were not exercised.\n")
    }
  }
  quit(status = 0)
}

cat("\n")
cat(strrep("=", 72), "\n", sep = "")
cat("  GPU-free R suites: ", length(failed), " of ", n_total, " FAILED",
    if (total_skips > 0L) sprintf("  (%d skipped)", total_skips) else "",
    "\n", sep = "")
cat(strrep("=", 72), "\n", sep = "")
for (nm in names(results)) {
  cat(suite_line(nm, if (results[[nm]]$code == 0L) "ok" else "FAIL"))
}
cat("\n")
cat("Re-run one suite on its own, from the repo root:\n")
cat("  Rscript ", failed[1], "\n", sep = "")
cat("\n")
cat("That form stops at the first failing group. To see every failure at once:\n")
cat("  Rscript -e 'testthat::set_max_fails(Inf); testthat::test_file(\"",
    failed[1], "\")'\n", sep = "")
cat("  -- but note test_file() chdirs into the test's directory, so a suite that\n")
cat("  resolves its source through a relative path may then fall through to the\n")
cat("  hardcoded absolute candidate. Trust the plain form for pass/fail.\n")
cat("\n")
cat("Do not read a suite's trailing \"All ... tests passed\" line as a verdict --\n")
cat("those cat() calls are unconditional. The exit status above is the verdict.\n")
quit(status = 1)

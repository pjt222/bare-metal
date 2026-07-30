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
# script, so the unconditional cat("All ... tests passed.") that each suite ends
# with (test_parser.R:178, test_meta.R:192, test_bench_all.R:507) is never
# reached. Under test_file() it prints even when every group in the file errored.
# Either way the verdict here comes from the child's exit status, never from what
# the child wrote.
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
# Usage:
#   Rscript scripts/audit/run_r_tests.R           # run every suite, report, exit 0/1
#   Rscript scripts/audit/run_r_tests.R --list    # print the discovered suites, run nothing
#   Rscript scripts/audit/run_r_tests.R --quiet   # only print on failure
#
# Exit codes:
#   0  every suite passed
#   1  at least one suite failed, errored, or could not be discovered

args  <- commandArgs(trailingOnly = TRUE)
quiet <- "--quiet" %in% args
list_only <- "--list" %in% args

unknown <- setdiff(args, c("--quiet", "--list"))
if (length(unknown)) {
  cat("run_r_tests.R: unknown argument(s): ", paste(unknown, collapse = " "), "\n", sep = "")
  cat("Usage: Rscript scripts/audit/run_r_tests.R [--list] [--quiet]\n")
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
# tests/**/test_*.R is picked up the day it lands.
suites <- list.files(
  file.path(repo_root, "tests"),
  pattern    = "^test_.*\\.R$",
  recursive  = TRUE,
  full.names = TRUE)
suites <- sort(suites)

# The cuasmR package carries its own testthat suite. It is GPU-free and the
# package is already on the library path (scripts/bench/bench_regress.R uses
# cuasmR::), so there is no extra install cost to running it here.
cuasmr_tests <- file.path(repo_root, "R", "cuasmR", "tests", "testthat")
has_cuasmr <- dir.exists(cuasmr_tests) &&
  length(list.files(cuasmr_tests, pattern = "^test-.*\\.R$")) > 0L

rel <- function(p) sub(paste0("^", repo_root, "/"), "", p)

if (length(suites) == 0L && !has_cuasmr) {
  cat("run_r_tests.R: no GPU-free R suites discovered under ", rel(file.path(repo_root, "tests")), "\n", sep = "")
  cat("This is a discovery failure, not an empty-but-healthy repo -- failing loudly\n")
  cat("rather than reporting a vacuous pass.\n")
  quit(status = 1)
}

n_total <- length(suites) + as.integer(has_cuasmr)

if (list_only) {
  cat("GPU-free R suites (", n_total, "):\n", sep = "")
  for (s in suites) cat("  ", rel(s), "\n", sep = "")
  if (has_cuasmr) cat("  ", rel(cuasmr_tests), " (cuasmR package suite)\n", sep = "")
  quit(status = 0)
}

if (!requireNamespace("testthat", quietly = TRUE)) {
  cat("run_r_tests.R: {testthat} is not installed -- cannot run the GPU-free suites.\n")
  cat("Install it with: Rscript -e 'renv::restore()'  (or make setup)\n")
  quit(status = 1)
}

# ---- run -------------------------------------------------------------------

rscript <- file.path(R.home("bin"), "Rscript")

run_child <- function(cmd_args, label) {
  if (!quiet) {
    cat("\n", strrep("-", 72), "\n", sep = "")
    cat("  ", label, "\n", sep = "")
    cat(strrep("-", 72), "\n", sep = "")
  }
  started <- Sys.time()
  # stdout/stderr inherit so a two-minute suite shows progress rather than
  # going silent. testthat writes its whole report to stderr, so never redirect
  # it away. The exit status is the verdict.
  code <- system2(rscript, cmd_args)
  list(code = as.integer(code),
       secs = as.numeric(difftime(Sys.time(), started, units = "secs")))
}

results <- list()

# Children run from the repo root so each suite's *relative* source candidate
# resolves. See the test_file() note in the header -- this is load-bearing.
old_wd <- setwd(repo_root)
on.exit(setwd(old_wd), add = TRUE)

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
total_secs <- sum(vapply(results, function(r) r$secs, numeric(1)))

if (length(failed) == 0L) {
  if (!quiet) {
    cat("\n")
    cat(strrep("=", 72), "\n", sep = "")
    cat("  GPU-free R suites: ", n_total, "/", n_total, " PASSED",
        sprintf("  (%.0f s)", total_secs), "\n", sep = "")
    cat(strrep("=", 72), "\n", sep = "")
    for (nm in names(results)) {
      cat(sprintf("  ok  %-52s %5.0f s\n", nm, results[[nm]]$secs))
    }
  }
  quit(status = 0)
}

cat("\n")
cat(strrep("=", 72), "\n", sep = "")
cat("  GPU-free R suites: ", length(failed), " of ", n_total, " FAILED\n", sep = "")
cat(strrep("=", 72), "\n", sep = "")
for (nm in names(results)) {
  cat(sprintf("  %-3s %-52s %5.0f s\n",
              if (results[[nm]]$code == 0L) "ok" else "FAIL",
              nm, results[[nm]]$secs))
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

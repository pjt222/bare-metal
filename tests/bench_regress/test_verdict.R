# tests/bench_regress/test_verdict.R
#
# GPU-free tests for the run verdict (#176) and the run record (#186) in
# scripts/bench/bench_regress.R.
#
# The bug the verdict groups pin: it branched on `regressions > 0L` alone, so a
# run in which every config SKIPPED printed "PASSED -- all benchmarks within
# tolerance" and exited 0 having measured nothing. It did that on five
# consecutive real pushes at 7 of 7 skipped, and the pre-push hook rendered it
# as a green "All benchmarks within tolerance. Push allowed."
#
# The bug the record groups pin: a push was rejected by a measured regression on
# 2026-07-31 and which config regressed could not be recovered, because the gate
# printed its verdicts and kept nothing.
#
# Two layers, deliberately. summarise_verdict() is a pure function of four
# counters and is tested directly. But a pure function can be perfect and never
# consulted -- #176 was a wiring bug -- so the end-to-end groups run the real
# script as a subprocess against a throwaway repo fixture and assert on its exit
# code, its output and the record it wrote. Still no GPU, no nvcc, no corpus.
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

.fixture_root <- function(throughput = NULL, baseline_gflops = 1000,
                          clock_lock = NULL) {
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

  cfg <- list(ms = 1.0, gflops = baseline_gflops)
  # A clock_lock entry routes the config down the clock-lock family of emit()
  # call sites instead of the direct one -- the branch three of the seven real
  # configs take on every ordinary push.
  if (!is.null(clock_lock)) cfg$clock_lock <- clock_lock

  baselines <- list(
    recorded_date = "fixture",
    platform = "fixture",
    default_valid_when = list(require_no_throttle = FALSE),
    kernels = list(`kernels/fixture/fake.cu` = list(exe = exe, `1_2` = cfg)))
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
       text = paste(out, collapse = "\n"),
       record = file.path(root, "results", "bench_regress", "gate_runs.jsonl"))
}

# Rows of the run record, in write order.
.record_rows <- function(run) {
  expect_true(file.exists(run$record))
  lapply(readLines(run$record, warn = FALSE), jsonlite::fromJSON,
         simplifyVector = FALSE)
}

# Each distinct scenario is run ONCE and its result shared by every group that
# asserts on it. Written the obvious way -- one .run_gate() per test_that -- this
# suite started twelve child R processes and cost 79 s of a blocking pre-push
# gate, most of it R startup on the 9p mount, re-deriving results it already had.
# Scenarios are keyed by name; the groups below read, they do not run.
.scenarios <- local({
  cache <- list()
  function(name) {
    if (!is.null(cache[[name]])) return(cache[[name]])
    run <- switch(
      name,
      skipped    = .run_gate(.fixture_root(throughput = NULL)),
      measured   = .run_gate(.fixture_root(throughput = 1000)),
      regressed  = .run_gate(.fixture_root(throughput = 500)),
      clocklock  = .run_gate(.fixture_root(throughput = 1000,
                                           clock_lock = 1605)),
      stop("unknown scenario: ", name))
    cache[[name]] <<- run
    run
  }
})

test_that("end to end: a run that skips everything exits 2 and says INCONCLUSIVE", {
  # This is #176 itself. Against the code as it was, this exits 0 and prints
  # "PASSED -- all benchmarks within tolerance".
  r <- .scenarios("skipped")
  expect_equal(r$status, 2L)
  expect_match(r$text, "INCONCLUSIVE", fixed = TRUE)
  expect_false(grepl("within tolerance", r$text, fixed = TRUE))
})

test_that("end to end: configs of an unbuilt kernel stay in the denominator", {
  # The emptier version of the same bug: the missing-executable branch used to
  # skip the whole kernel before any counter moved, reporting `Total: 0`.
  r <- .scenarios("skipped")
  expect_match(r$text, "Total: 1", fixed = TRUE)
  expect_match(r$text, "Measured: 0", fixed = TRUE)
  expect_match(r$text, "0 of 1", fixed = TRUE)
})

test_that("end to end: a config that measures cleanly exits 0 and is counted", {
  r <- .scenarios("measured")
  expect_equal(r$status, 0L)
  expect_match(r$text, "Measured: 1", fixed = TRUE)
  expect_match(r$text, "1 of 1 config(s) measured", fixed = TRUE)
})

test_that("end to end: a measured regression exits 1", {
  # Half of baseline, far outside the 10% default tolerance.
  r <- .scenarios("regressed")
  expect_equal(r$status, 1L)
  expect_match(r$text, "FAILED", fixed = TRUE)
  expect_match(r$text, "Measured: 1", fixed = TRUE)
})

# ---- the run record (#186) ---------------------------------------------
#
# A push was rejected by a measured regression on 2026-07-31 and which config
# regressed could not be recovered: stdout was the only record, and a re-run ten
# minutes later passed. These pin the record that now outlives the terminal.

test_that("every run leaves a record, including one that measured nothing", {
  rows <- .record_rows(.scenarios("skipped"))
  expect_equal(length(rows), 2L)                       # 1 config + 1 summary
  expect_equal(vapply(rows, function(r) r$type, ""),
               c("config", "run_summary"))
})

test_that("the record names the configs behind a FAILED verdict", {
  # The whole point of #186: after the terminal is gone, this is what says
  # which kernel and which config were responsible.
  rows <- .record_rows(.scenarios("regressed"))
  summary_row <- rows[[length(rows)]]
  expect_equal(summary_row$verdict, "FAILED")
  expect_equal(summary_row$exit, 1L)
  expect_equal(length(summary_row$failed), 1L)
  expect_equal(summary_row$failed[[1]]$kernel, "kernels/fixture/fake.cu")
  expect_equal(summary_row$failed[[1]]$config, "1_2")
  expect_match(summary_row$failed[[1]]$msg, "REGRESSION", fixed = TRUE)
})

test_that("a measured config records its number, its baseline and its tolerance", {
  rows <- .record_rows(.scenarios("measured"))
  cfg <- rows[[1]]
  expect_equal(cfg$verdict, "OK")
  expect_true(cfg$measured)
  expect_equal(cfg$throughput, 1000)
  expect_equal(cfg$baseline_gflops, 1000)
  expect_equal(cfg$tolerance, 0.1)
  expect_equal(cfg$returncode, 0L)
  # The `meta` slot is always present, and in THIS harness it is always null:
  # the fixture's child process gets no GPU snapshot even on the GPU box.
  # Measured, not assumed -- and unexplained, so the digest is covered by the
  # direct meta_digest() groups below rather than by a conditional here that
  # would silently never run. See the harness note at the top of this section.
  expect_true("meta" %in% names(cfg))
})

# ---- the GPU digest, tested directly -----------------------------------
#
# meta_digest() is what makes a thermal false positive tellable from a real
# regression the morning after, so it needs real coverage. It cannot get that
# from the end-to-end fixture: the child process there returns no GPU snapshot,
# on CI and on the GPU box alike, for reasons not established (the capture works
# from a tempdir and works in the test parent, so it is neither cwd nor the
# LD_LIBRARY_PATH the script sets). Tested here against synthetic snapshots
# shaped like capture_gpu_state()'s output instead.

.fake_snapshot <- function(throttle = character(0)) {
  list(gpu = list(clock_sm = 1770, clock_mem = 7001, temp_c = 62,
                  power_w = 134.8, pstate = "P0", throttle = throttle),
       host = list(ac_state = "battery"))
}

test_that("meta_digest carries the fields a false positive is judged on", {
  d <- meta_digest(list(meta_pre = .fake_snapshot(),
                        meta_post = .fake_snapshot("SwPowerCap")))
  expect_equal(d$clock_sm, 1770)
  expect_equal(d$temp_c, 62)
  expect_equal(d$power_w, 134.8)
  expect_equal(d$pstate, "P0")
  expect_equal(d$ac_state, "battery")
})

test_that("meta_digest is NULL when there was no GPU snapshot", {
  # Every CI runner. The row must still be written, with a null meta.
  expect_null(meta_digest(list(meta_pre = NULL, meta_post = NULL)))
  expect_null(meta_digest(NULL))
})

test_that("throttle stays a JSON array at every length", {
  # auto_unbox would emit a bare string for a single reason and an array for
  # two, so a consumer indexing [0] would read "S" from "SwPowerCap" on
  # exactly the single-reason runs that are the common case.
  for (reasons in list(character(0), "SwPowerCap",
                       c("SwPowerCap", "SwThermalSlowdown"))) {
    d <- meta_digest(list(meta_pre = .fake_snapshot(reasons),
                          meta_post = .fake_snapshot(reasons)))
    json <- jsonlite::toJSON(d, auto_unbox = TRUE, na = "null", null = "null")
    parsed <- jsonlite::fromJSON(as.character(json), simplifyVector = FALSE)
    expect_true(is.list(parsed$throttle))
    expect_equal(length(parsed$throttle), length(reasons))
  }
})

test_that("a skipped config records why, not merely that", {
  rows <- .record_rows(.scenarios("skipped"))
  cfg <- rows[[1]]
  expect_equal(cfg$verdict, "SKIPPED")
  expect_false(cfg$measured)
  expect_match(cfg$msg, "executable not found", fixed = TRUE)
})

test_that("a clock-locked config skips, is counted, and records its lock", {
  # The clock-lock family of emit() call sites, which nothing else here
  # reaches: without --clock-locked these skip by design, and on this laptop
  # that is three of the seven real configs on every ordinary push (#156).
  r <- .scenarios("clocklock")
  expect_equal(r$status, 2L)                       # measured nothing
  expect_match(r$text, "clock_lock 1605 MHz", fixed = TRUE)
  cfg <- .record_rows(r)[[1]]
  expect_equal(cfg$verdict, "SKIPPED")
  expect_false(cfg$measured)
  expect_equal(cfg$clock_lock, 1605L)
  expect_null(cfg$clock_locked_arg)                # none was passed
})

test_that("the operator-visible summary lines are printed, not just recorded", {
  # Both were added for a human reading a rejected push: the recap of what
  # failed, and where the evidence went. Neither is covered by asserting on
  # the JSONL.
  failing <- .scenarios("regressed")
  expect_match(failing$text, "Configs behind this verdict:", fixed = TRUE)
  expect_match(failing$text, "kernels/fixture/fake.cu [1_2]", fixed = TRUE)
  expect_match(failing$text, "Record: ", fixed = TRUE)
  expect_match(.scenarios("measured")$text, "Record: ", fixed = TRUE)
})

test_that("the run summary counts the rows it is summarising", {
  summary_row <- utils::tail(.record_rows(.scenarios("measured")), 1)[[1]]
  expect_equal(summary_row$type, "run_summary")
  expect_equal(summary_row$config_rows_written, 1L)
  expect_equal(summary_row$config_rows_attempted, 1L)
  # git_head identifies the tree that produced the numbers -- a store spanning
  # weeks can otherwise say a config regressed but not against what. The slot
  # is always present; its VALUE is null here because the fixture repo is
  # marked by a bare renv.lock and has no .git for `rev-parse` to read. A real
  # run records a short sha, and the gate must not fail for want of one.
  expect_true("git_head" %in% names(summary_row))
})

test_that("the record is append-only across runs, with distinct run ids", {
  # Two runs against one fixture: the second must not truncate the first.
  # A store that overwrites answers "what happened just now" and nothing else,
  # which is the failure being fixed.
  root <- .fixture_root(throughput = NULL)
  first <- .record_rows(.run_gate(root))
  both  <- .record_rows(.run_gate(root))
  expect_equal(length(both), 2L * length(first))
  ids <- unique(vapply(both, function(r) r$run_id, ""))
  expect_equal(length(ids), 2L)
})

test_that("a record that cannot be written does not change the verdict", {
  # Recording is evidence, not enforcement. On a read-only checkout the gate
  # must still reach the same conclusion and the same exit code -- and say
  # plainly that it kept nothing.
  root <- .fixture_root(throughput = NULL)
  dir.create(file.path(root, "results", "bench_regress"), recursive = TRUE)
  Sys.chmod(file.path(root, "results", "bench_regress"), "0500")
  on.exit(Sys.chmod(file.path(root, "results", "bench_regress"), "0755"),
          add = TRUE)
  skip_if(file.access(file.path(root, "results", "bench_regress"), 2L) == 0L,
          "filesystem ignores the read-only bit (running as root?)")

  r <- .run_gate(root)
  expect_equal(r$status, 2L)                       # unchanged by the failure
  expect_match(r$text, "INCONCLUSIVE", fixed = TRUE)
  expect_match(r$text, "NOT WRITTEN", fixed = TRUE)
})

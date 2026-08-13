# tests/bench_regress/test_meta.R
#
# Tests for the GPU/host metadata capture in cuasmR --
# capture_gpu_state(), classify_meta(), decode_throttle(), summarise_meta()
# (R/cuasmR/R/bench_meta.R). Most tests use canned data so they pass on any
# host (CI runners without nvidia-smi included). A live-capture smoke test
# runs only when nvidia-smi is on PATH.
#
# Run with:
#   Rscript tests/bench_regress/test_meta.R
#   (or)  Rscript -e 'testthat::test_file("tests/bench_regress/test_meta.R")'

library(testthat)

# The subject is the package, so attach it directly (#173). This used to
# source scripts/bench/bench_meta.R, which is a compatibility shim whose own
# header says it is slated for removal once callers use library(cuasmR) --
# a gated suite was the thing keeping it alive, and deleting the shim would
# have turned the gate red for no reason. The shim's entire body is
# `library(cuasmR)`; nothing is lost. That source() also carried a hardcoded
# /mnt/d/dev/p/bare-metal/... fallback, which went with it.
suppressMessages(library(cuasmR))

# ---- decode_throttle ----------------------------------------------------

test_that("decode_throttle: 0x0 -> empty (no throttle)", {
  expect_equal(decode_throttle("0x0000000000000000"), character(0))
  expect_equal(decode_throttle("0x0"), character(0))
})

test_that("decode_throttle: 0x1 -> GpuIdle", {
  expect_equal(decode_throttle("0x0000000000000001"), "GpuIdle")
})

test_that("decode_throttle: 0x4 -> SwPowerCap (real-world case)", {
  expect_equal(decode_throttle("0x0000000000000004"), "SwPowerCap")
})

test_that("decode_throttle: 0x44 -> HwThermalSlowdown + SwPowerCap (combined)", {
  result <- decode_throttle("0x0000000000000044")
  expect_setequal(result, c("SwPowerCap", "HwThermalSlowdown"))
})

test_that("decode_throttle: NULL/empty input -> empty", {
  expect_equal(decode_throttle(NULL), character(0))
  expect_equal(decode_throttle(""), character(0))
})

test_that("decode_throttle: unparseable mask -> NA, NOT empty", {
  # BEHAVIOUR CHANGE (#208). This group previously asserted character(0)
  # for an unparseable mask, which pinned the defect rather than the
  # contract: character(0) is also the value for "no throttle", so a mask
  # the decoder could not read was reported as a clean GPU and the sample
  # was compared against a baseline.
  #
  # The two answers must stay distinguishable. character(0) says "the GPU
  # was not throttled" and licenses a comparison; NA says "we could not
  # tell" and must not. classify_meta surfaces the NA through setdiff() as
  # an unrecognised reason, so the sample is skipped -- the safe direction.
  expect_equal(decode_throttle("not_hex"), NA_character_)
  expect_equal(decode_throttle("0xZZZZ"), NA_character_)
  # Absent input is still "nothing to report", not a parse failure.
  expect_equal(decode_throttle(NULL), character(0))
  expect_equal(decode_throttle(""), character(0))
})

# ---- classify_meta ------------------------------------------------------

# Build a synthetic snapshot for tests so we don't depend on a live GPU.
.fake_state <- function(throttle_hex = "0x0000000000000000",
                        clock_sm = 1700, clock_mem = 7001,
                        temp_c = 50, power_w = 80, pstate = "P0",
                        ac_state = "ac",
                        load_1m = 0.5) {
  list(
    gpu = list(
      clock_sm     = clock_sm,
      clock_mem    = clock_mem,
      temp_c       = temp_c,
      power_w      = power_w,
      pstate       = pstate,
      throttle_hex = throttle_hex,
      throttle     = decode_throttle(throttle_hex),
      util_gpu     = 100,
      util_mem     = 90
    ),
    host = list(
      loadavg  = list(load_1m = load_1m, load_5m = load_1m, load_15m = load_1m),
      ac_state = ac_state
    ),
    iso_time = "2026-05-10T12:00:00+0000"
  )
}

test_that("classify_meta: clean state passes default policy", {
  s <- .fake_state(throttle_hex = "0x0")
  cls <- classify_meta(s, s)
  expect_true(cls$ok)
  expect_length(cls$reasons, 0)
})

test_that("classify_meta: GpuIdle is allowed by default (idle between launches)", {
  s <- .fake_state(throttle_hex = "0x1")  # GpuIdle
  cls <- classify_meta(s, s)
  expect_true(cls$ok)
})

test_that("classify_meta: SwPowerCap is rejected as unfair", {
  s <- .fake_state(throttle_hex = "0x4")  # SwPowerCap
  cls <- classify_meta(s, s)
  expect_false(cls$ok)
  expect_match(paste(cls$reasons, collapse = " "),
               "SwPowerCap", fixed = TRUE)
})

test_that("classify_meta: HwThermalSlowdown is rejected as unfair", {
  s <- .fake_state(throttle_hex = "0x40")  # HwThermalSlowdown
  cls <- classify_meta(s, s)
  expect_false(cls$ok)
  expect_match(paste(cls$reasons, collapse = " "),
               "HwThermalSlowdown", fixed = TRUE)
})

test_that("classify_meta: pre-only throttle is also flagged", {
  pre  <- .fake_state(throttle_hex = "0x40")  # HwThermalSlowdown during warmup
  post <- .fake_state(throttle_hex = "0x0")
  cls <- classify_meta(pre, post)
  expect_false(cls$ok)
})

test_that("classify_meta: min_clock_sm enforces a floor", {
  s_fast <- .fake_state(clock_sm = 1700)
  s_slow <- .fake_state(clock_sm = 900)
  expect_true(classify_meta(s_fast, s_fast,
                             list(min_clock_sm = 1500))$ok)
  expect_false(classify_meta(s_slow, s_slow,
                              list(min_clock_sm = 1500))$ok)
})

test_that("classify_meta: max_temp_c enforces a ceiling", {
  s_cool <- .fake_state(temp_c = 50)
  s_hot  <- .fake_state(temp_c = 85)
  expect_true(classify_meta(s_cool, s_cool, list(max_temp_c = 75))$ok)
  expect_false(classify_meta(s_hot,  s_hot,  list(max_temp_c = 75))$ok)
})

test_that("classify_meta: require_ac flags battery operation", {
  s_ac  <- .fake_state(ac_state = "ac")
  s_bat <- .fake_state(ac_state = "battery")
  expect_true(classify_meta(s_ac,  s_ac,  list(require_ac = TRUE))$ok)
  expect_false(classify_meta(s_bat, s_bat, list(require_ac = TRUE))$ok)
})

test_that("classify_meta: NULL state -> ok=NA (graceful no-GPU)", {
  cls <- classify_meta(NULL, NULL)
  expect_true(is.na(cls$ok))
  expect_match(cls$summary, "unavailable", fixed = TRUE)
})

test_that("classify_meta: multiple reasons accumulate", {
  s <- .fake_state(throttle_hex = "0x4", clock_sm = 800,
                   temp_c = 90, ac_state = "battery")
  cls <- classify_meta(s, s, list(min_clock_sm = 1500, max_temp_c = 75,
                                   require_ac = TRUE))
  expect_false(cls$ok)
  expect_gte(length(cls$reasons), 4)
})

test_that("classify_meta: summary string includes key fields", {
  s <- .fake_state(clock_sm = 1700, clock_mem = 7000, temp_c = 60,
                   power_w = 90, pstate = "P0", throttle_hex = "0x0")
  cls <- classify_meta(s, s)
  expect_match(cls$summary, "1700", fixed = TRUE)
  expect_match(cls$summary, "60",   fixed = TRUE)
  expect_match(cls$summary, "P0",   fixed = TRUE)
  expect_match(cls$summary, "throttle=none", fixed = TRUE)
})

test_that("classify_meta: summary carries the power triple (installed build)", {
  # This suite attaches the INSTALLED cuasmR, so it is the only place that
  # notices a bench_meta.R edit which was never reinstalled (#207). The
  # package's own test-power_policy.R runs under test_local() against the
  # source tree and passes regardless of what is installed -- so without
  # this assertion, `make test-r` stays green while every gate row writes
  # power_limit_w = null.
  s <- .fake_state(clock_sm = 1700, temp_c = 60, power_w = 40)
  s$gpu$power_limit_w         <- 50
  s$gpu$power_limit_default_w <- 115
  s$gpu$power_limit_max_w     <- 150
  s$gpu$power_below_default   <- TRUE
  s$gpu$power_below_max       <- TRUE
  expect_match(classify_meta(s, s)$summary, "POWER-LIMIT=50/115/150W",
               fixed = TRUE)
})

# ---- summarise_meta -----------------------------------------------------

test_that("summarise_meta: NULL inputs -> '(no GPU meta)'", {
  expect_equal(summarise_meta(NULL, NULL), "(no GPU meta)")
})

test_that("summarise_meta: clean state produces a one-line summary", {
  s <- .fake_state()
  out <- summarise_meta(s, s)
  expect_match(out, "throttle=none", fixed = TRUE)
})

# ---- live capture (only if nvidia-smi is reachable) --------------------

test_that("capture_gpu_state: live capture round-trips on real hardware", {
  skip_if_not(file.exists("/usr/lib/wsl/lib/nvidia-smi") ||
              file.exists("/usr/bin/nvidia-smi"),
              "nvidia-smi not available")
  s <- capture_gpu_state()
  if (is.null(s)) skip("nvidia-smi found but driver communication failed")
  expect_true(is.numeric(s$gpu$clock_sm))
  expect_true(s$gpu$clock_sm > 0)
  expect_true(is.character(s$gpu$pstate))
  expect_true(s$host$ac_state %in% c("ac", "battery", "unknown"))
})

# No trailing "all tests passed" line. It was unconditional, so under
# testthat::test_file() it printed even when every group above had errored --
# which is how the retired tests/bench_regress/test_parser.R advertised
# "All bench_regress parser tests passed." for 58 days while all 14 of its
# groups were failing (#171). The verdict comes from the exit status.

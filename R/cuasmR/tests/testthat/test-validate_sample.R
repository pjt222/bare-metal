# Differential / characterization test for validate_sample (issue #134,
# Phase 3). Proves the unified per-sample verdict reproduces the
# valid/invalid DECISION of each of the three original inline validators.
# Reason strings are intentionally canonicalised (grid_collect histograms
# them tolerantly), so assertions are on $ok only.
#
# The three originals are inlined VERBATIM below as oracles. The load-
# bearing case is pre-throttled / post-clean: grid validated on POST only
# (throttle_str(post)), rebaseline/bench_regress on PRE+POST via
# classify_meta — so the unified validate_sample must reproduce both, and
# grid does so by passing its post snapshot as BOTH pre and post.

mk_state <- function(throttle = "GpuIdle", clock_sm = 1700, temp_c = 50,
                     ac = "ac", power_w = 50, clock_mem = 7000, pstate = "P0") {
  list(
    gpu = list(clock_sm = clock_sm, clock_mem = clock_mem, temp_c = temp_c,
               power_w = power_w, pstate = pstate, throttle = throttle,
               throttle_hex = "0x0", util_gpu = 100, util_mem = 50),
    host = list(loadavg = NULL, ac_state = ac, gpu_mode = "unknown"),
    iso_time = "t")
}

# ---- Oracle: grid_measure.R inline (POST-only throttle, lines 286-299) --
oracle_grid <- function(rc, throughput, post, clk_tgt = NULL, band = 30L) {
  if (rc != 0L) return(FALSE)
  if (is.na(throughput)) return(FALSE)
  thr_reasons <- setdiff(post$gpu$throttle, "GpuIdle")
  thr <- if (length(thr_reasons) == 0L) "none" else paste(thr_reasons, collapse = ",")
  if (thr != "none") return(FALSE)
  clk_obs <- as.integer(post$gpu$clock_sm)
  if (!is.null(clk_tgt) && !is.na(clk_obs)) {
    if (clk_obs < clk_tgt - band || clk_obs > clk_tgt + band) return(FALSE)
  }
  TRUE
}

# ---- Oracle: rebaseline_measure.R classify_sample (PRE+POST, min_clk) ---
oracle_rebaseline <- function(rc, throughput, pre, post, min_clk = 1300L) {
  if (rc != 0L) return(FALSE)                       # crash
  if (is.na(throughput)) return(FALSE)              # parse-fail
  meta <- classify_meta(pre, post,
                        list(require_no_throttle = TRUE, allow_throttle = c("GpuIdle")))
  if (isFALSE(meta$ok)) return(FALSE)               # throttled
  clk <- post$gpu$clock_sm
  if (is.na(clk) || clk < min_clk) return(FALSE)    # cold-clock
  TRUE
}

# ---- Oracle: bench_regress.R measure_clock_locked (PRE+POST, band) ------
oracle_bench_regress <- function(rc, throughput, pre, post, valid_when,
                                 clock_lock, band = 30L) {
  if (!is.null(rc) && rc != 0L) return(FALSE)       # crash
  if (is.null(throughput)) return(FALSE)            # parse-fail
  cls <- classify_meta(pre, post, valid_when)
  if (isFALSE(cls$ok)) return(FALSE)                # unfair
  if (is.na(cls$ok)) return(FALSE)                  # no-gpu-meta
  clk <- post$gpu$clock_sm
  lo <- clock_lock - band; hi <- clock_lock + band
  if (is.null(clk) || is.na(clk) || clk < lo || clk > hi) return(FALSE)
  TRUE
}

# Scenario grid: each row = (rc, throughput, pre throttle, post throttle, post clk)
scen <- list(
  list(rc = 0L,   tp = 100, pre = "GpuIdle",    post = "GpuIdle",   clk = 1700),
  list(rc = 127L, tp = 100, pre = "GpuIdle",    post = "GpuIdle",   clk = 1700), # crash
  list(rc = 0L,   tp = NA,  pre = "GpuIdle",    post = "GpuIdle",   clk = 1700), # parse-fail
  list(rc = 0L,   tp = 100, pre = "GpuIdle",    post = "SwPowerCap",clk = 1700), # post throttled
  list(rc = 0L,   tp = 100, pre = "SwPowerCap", post = "GpuIdle",   clk = 1700), # PRE throttled, post clean (discriminating)
  list(rc = 0L,   tp = 100, pre = "GpuIdle",    post = "GpuIdle",   clk = 1250), # low clock
  list(rc = 0L,   tp = 100, pre = "GpuIdle",    post = "GpuIdle",   clk = 1300), # exactly min_clk
  list(rc = 0L,   tp = 100, pre = "GpuIdle",    post = "GpuIdle",   clk = 1299), # just below min_clk
  list(rc = 0L,   tp = 100, pre = "GpuIdle",    post = "GpuIdle",   clk = NA_integer_) # NA clock
)

test_that("validate_sample reproduces grid (post-only throttle) decision", {
  for (s in scen) {
    post <- mk_state(throttle = s$post, clock_sm = s$clk)
    clk_obs <- as.integer(post$gpu$clock_sm)
    for (clk_tgt in list(NULL, 1605L)) {
      want <- oracle_grid(s$rc, s$tp, post, clk_tgt = clk_tgt)
      # Replicate grid's actual call: validate on POST only (pass post as
      # both pre and post); apply the band only when clk_tgt is set AND the
      # observed clock is non-NA (grid's original guard skips the band on
      # an NA clock).
      apply_band <- !is.null(clk_tgt) && !is.na(clk_obs)
      got <- validate_sample(s$rc, s$tp, post, post,
                             valid_when = list(allow_throttle = c("GpuIdle")),
                             clock_band = if (apply_band)
                                            c(clk_tgt - 30L, clk_tgt + 30L) else NULL)$ok
      expect_equal(got, want,
                   info = sprintf("grid post=%s clk=%s clk_tgt=%s",
                                  s$post, as.character(s$clk),
                                  if (is.null(clk_tgt)) "native" else clk_tgt))
    }
  }
})

test_that("validate_sample reproduces rebaseline (pre+post throttle, min_clk floor) decision", {
  for (s in scen) {
    pre  <- mk_state(throttle = s$pre)
    post <- mk_state(throttle = s$post, clock_sm = s$clk)
    want <- oracle_rebaseline(s$rc, s$tp, pre, post, min_clk = 1300L)
    got <- validate_sample(s$rc, s$tp, pre, post,
                           valid_when = list(allow_throttle = c("GpuIdle"),
                                             min_clock_sm = 1300L),
                           clock_band = NULL)$ok
    expect_equal(got, want,
                 info = sprintf("rebaseline pre=%s post=%s clk=%d", s$pre, s$post, s$clk))
  }
})

test_that("validate_sample reproduces bench_regress (pre+post, two-sided band) decision", {
  vw <- list(require_no_throttle = TRUE, allow_throttle = c("GpuIdle"))
  for (s in scen) {
    pre  <- mk_state(throttle = s$pre)
    post <- mk_state(throttle = s$post, clock_sm = s$clk)
    rc <- if (s$rc == 0L) 0L else s$rc
    tp <- if (is.na(s$tp)) NULL else s$tp   # bench_regress uses is.null(throughput)
    want <- oracle_bench_regress(rc, tp, pre, post, vw, clock_lock = 1605L)
    got <- validate_sample(rc, if (is.null(tp)) NA_real_ else tp, pre, post,
                           valid_when = vw,
                           clock_band = c(1605L - 30L, 1605L + 30L))$ok
    expect_equal(got, want,
                 info = sprintf("bench_regress pre=%s post=%s clk=%d", s$pre, s$post, s$clk))
  }
})

test_that("the discriminating case: pre-throttled/post-clean diverges by caller", {
  pre  <- mk_state(throttle = "SwPowerCap")
  post <- mk_state(throttle = "GpuIdle", clock_sm = 1700)
  # grid (post-only) -> VALID
  expect_true(validate_sample(0L, 100, post, post,
                              valid_when = list(allow_throttle = c("GpuIdle")))$ok)
  # rebaseline/bench_regress (pre+post) -> INVALID
  expect_false(validate_sample(0L, 100, pre, post,
                               valid_when = list(allow_throttle = c("GpuIdle")))$ok)
})

test_that("clock band edges are inclusive; NA clock in a band is rejected", {
  post_lo <- mk_state(clock_sm = 1575); post_hi <- mk_state(clock_sm = 1635)
  post_under <- mk_state(clock_sm = 1574); post_na <- mk_state(clock_sm = NA_integer_)
  band <- c(1575L, 1635L)
  expect_true (validate_sample(0L, 100, post_lo, post_lo, clock_band = band)$ok)
  expect_true (validate_sample(0L, 100, post_hi, post_hi, clock_band = band)$ok)
  expect_false(validate_sample(0L, 100, post_under, post_under, clock_band = band)$ok)
  expect_false(validate_sample(0L, 100, post_na, post_na, clock_band = band)$ok)
})

test_that("collect_valid_samples loops until N valid or hits the attempt cap", {
  # Stub: every 3rd attempt is valid, others reject. GPU-free.
  attempt_seen <- 0L
  sample_fn <- function() { attempt_seen <<- attempt_seen + 1L; attempt_seen }
  validate_fn <- function(s) if (s %% 3L == 0L) list(ok = TRUE, reason = NA_character_)
                             else list(ok = FALSE, reason = "stub-reject")
  r <- collect_valid_samples(sample_fn, validate_fn, n_valid = 2L, max_attempts = 20L)
  expect_true(r$complete)
  expect_length(r$samples, 2L)
  expect_equal(r$samples, list(3L, 6L))     # full sample objects kept
  expect_equal(r$attempts, 6L)              # stopped as soon as 2nd valid found
  expect_equal(r$rejected, rep("stub-reject", 4L))

  # Never enough valid -> incomplete, capped at max_attempts.
  attempt_seen <<- 0L
  never_ok <- function(s) list(ok = FALSE, reason = "nope")
  r2 <- collect_valid_samples(sample_fn, never_ok, n_valid = 5L, max_attempts = 7L)
  expect_false(r2$complete)
  expect_length(r2$samples, 0L)
  expect_equal(r2$attempts, 7L)
})

test_that("report_median_metrics matches manual median/min/max", {
  tput <- c(100, 110, 90, 105, 95)
  ms   <- c(1.0, 1.1, 0.9, 1.05, 0.95)
  clk  <- c(1600L, 1610L, 1590L, 1605L, 1595L)
  r <- report_median_metrics(tput, ms, clk)
  expect_equal(r$median_throughput, stats::median(tput))
  expect_equal(r$median_ms, stats::median(ms))
  expect_equal(r$tput_lo, 90); expect_equal(r$tput_hi, 110)
  expect_equal(r$clk_lo, 1590L); expect_equal(r$clk_hi, 1610L)
  expect_equal(r$n, 5L)
  # ms with NA uses na.rm (bench_regress behaviour)
  expect_equal(report_median_metrics(c(100, 200), c(1.0, NA))$median_ms, 1.0)
})

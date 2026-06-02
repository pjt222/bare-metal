# Unit test for check_regression (issue #134, Phase 5). GPU-free: the
# meta snapshots are hand-built, so the fairness (SKIPPED) path is
# exercised without a card.

st <- function(throttle = "GpuIdle") {
  list(gpu = list(clock_sm = 1700, clock_mem = 7000, temp_c = 50,
                  power_w = 50, pstate = "P0", throttle = throttle,
                  throttle_hex = "0x0", util_gpu = 100, util_mem = 50),
       host = list(loadavg = NULL, ac_state = "ac", gpu_mode = "unknown"),
       iso_time = "t")
}
clean <- st("GpuIdle")

test_that("ratio buckets: OK / IMPROVED / REGRESSION", {
  base <- list(gflops = 1000)
  expect_match(check_regression(list(throughput = 1090, unit = "GFLOPS",
                                     meta_pre = clean, meta_post = clean),
                                base, 0.10)$msg, "^OK")
  imp <- check_regression(list(throughput = 1200, unit = "GFLOPS",
                               meta_pre = clean, meta_post = clean), base, 0.10)
  expect_false(imp$is_reg); expect_match(imp$msg, "^IMPROVED")
  reg <- check_regression(list(throughput = 800, unit = "GFLOPS",
                               meta_pre = clean, meta_post = clean), base, 0.10)
  expect_true(reg$is_reg); expect_match(reg$msg, "^REGRESSION")
})

test_that("crash and no-data are regressions", {
  expect_true(check_regression(list(returncode = 1L), list(gflops = 1000), 0.10)$is_reg)
  expect_match(check_regression(list(returncode = 134L), list(gflops = 1), 0.1)$msg, "CRASH")
  nd <- check_regression(list(throughput = 0, unit = "GFLOPS",
                              meta_pre = clean, meta_post = clean),
                         list(gflops = 1000), 0.10)
  expect_true(nd$is_reg); expect_match(nd$msg, "NO_DATA")
})

test_that("unfair GPU state -> SKIPPED, not REGRESSION", {
  bad <- st("SwPowerCap")
  r <- check_regression(list(throughput = 500, unit = "GFLOPS",  # would be a regression
                             meta_pre = clean, meta_post = bad),
                        list(gflops = 1000), 0.10)
  expect_false(r$is_reg)
  expect_true(isTRUE(r$skipped))
  expect_match(r$msg, "^SKIPPED")
})

test_that("unit selects the matching baseline field (TOPS vs GFLOPS)", {
  base <- list(tops = 500, gflops = 9999)
  r <- check_regression(list(throughput = 510, unit = "TOPS",
                             meta_pre = clean, meta_post = clean), base, 0.10)
  expect_match(r$msg, "^OK")
  expect_match(r$msg, "TOPS")
})

test_that("missing meta (off-GPU) skips the fairness check and compares anyway", {
  r <- check_regression(list(throughput = 1050, unit = "GFLOPS"),  # no meta_pre/post
                        list(gflops = 1000), 0.10)
  expect_match(r$msg, "^OK")
})

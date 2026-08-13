# Platform power-envelope tracking (issue #207).
#
# The motivating incident: on 2026-08-13 this machine ran with
# enforced.power.limit = 50 W against a 115 W default (Windows power
# overlay flipped to "Best power efficiency" on AC). conv2d measured 51%
# of its clean value, and NOTHING in the run record said the GPU had been
# capped -- power.draw only reports what the GPU drew, never what it was
# allowed to draw.

test_that(".power_capped detects a platform cap", {
  expect_true(cuasmR:::.power_capped(50, 115))
  expect_true(cuasmR:::.power_capped(90, 115))
})

test_that(".power_capped is FALSE at the full envelope", {
  expect_false(cuasmR:::.power_capped(115, 115))
  # A driver reporting the pair a hair apart is not a cap.
  expect_false(cuasmR:::.power_capped(114.5, 115))
})

test_that(".power_capped returns NA when it cannot tell, never FALSE", {
  # This is the load-bearing case. "We do not know whether this session
  # was capped" must not be recorded as "this session was not capped" --
  # only the second licenses a comparison against a baseline.
  expect_true(is.na(cuasmR:::.power_capped(NULL, 115)))
  expect_true(is.na(cuasmR:::.power_capped(50, NULL)))
  expect_true(is.na(cuasmR:::.power_capped(NA_real_, 115)))
  expect_true(is.na(cuasmR:::.power_capped(50, NA_real_)))
  # nvidia-smi reports "[N/A]" as a string when the field is unsupported.
  expect_true(is.na(cuasmR:::.power_capped("[N/A]", 115)))
  expect_true(is.na(cuasmR:::.power_capped(50, "[N/A]")))
  expect_true(is.na(cuasmR:::.power_capped(50, 0)))
})

test_that("classify_meta appends the cap to its summary only when capped", {
  mk <- function(capped) {
    list(gpu = list(clock_sm = 1770, clock_mem = 7001, temp_c = 60,
                    power_w = 65, pstate = "P0", throttle = "GpuIdle",
                    power_limit_w = if (capped) 50 else 115,
                    power_limit_default_w = 115,
                    power_capped = capped),
         host = list(ac_state = "ac", gpu_mode = "unknown"))
  }
  capped   <- classify_meta(mk(TRUE),  mk(TRUE))
  uncapped <- classify_meta(mk(FALSE), mk(FALSE))

  expect_match(capped$summary, "POWER-CAPPED=50/115W", fixed = TRUE)
  expect_false(grepl("POWER-CAPPED", uncapped$summary, fixed = TRUE))
})

test_that("classify_meta tolerates snapshots with no power-limit fields", {
  # Every pre-#207 caller builds snapshots without these fields; they must
  # keep working and must not gain a spurious POWER-CAPPED marker.
  old <- list(gpu = list(clock_sm = 1770, clock_mem = 7001, temp_c = 60,
                         power_w = 65, pstate = "P0", throttle = "GpuIdle"),
              host = list(ac_state = "ac", gpu_mode = "unknown"))
  res <- classify_meta(old, old)
  expect_true(res$ok)
  expect_false(grepl("POWER-CAPPED", res$summary, fixed = TRUE))
})

test_that("capture_power_policy never guesses when the host cannot be read", {
  # Off Windows/WSL there is no powershell.exe; the contract is explicit
  # NA + source="unavailable", not a fabricated default.
  pol <- capture_power_policy()
  expect_true(is.list(pol))
  expect_true(all(c("overlay_guid", "overlay_name", "ac_overlay_guid",
                    "dc_overlay_guid", "source") %in% names(pol)))
  if (identical(pol$source, "unavailable")) {
    expect_true(is.na(pol$overlay_guid))
    expect_true(is.na(pol$overlay_name))
  } else {
    expect_identical(pol$source, "windows-registry")
    expect_true(nzchar(pol$overlay_guid))
    expect_true(nzchar(pol$overlay_name))
  }
})

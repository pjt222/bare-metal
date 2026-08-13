# Platform power-envelope tracking (issue #207).
#
# The motivating incident: on 2026-08-13 this machine ran with
# enforced.power.limit = 50 W against a 115 W default and a 150 W VBIOS
# max. conv2d measured 51% of its clean value, and NOTHING in the run
# record said the GPU had been capped -- power.draw only reports what the
# GPU drew, never what it was allowed to draw.
#
# The reference matters as much as the reading. docs/benchmark_methodology.md
# records this machine's normal state as Current = Max = 150 W with the
# default at 115 W, so "below default" alone is blind to the documented
# 115 W fallback -- a 23% envelope cut. Both comparisons are kept.

test_that(".power_below detects a limit under its reference", {
  expect_true(cuasmR:::.power_below(50, 115))
  expect_true(cuasmR:::.power_below(90, 115))
  expect_true(cuasmR:::.power_below(115, 150))
})

test_that(".power_below is FALSE at or above the reference", {
  expect_false(cuasmR:::.power_below(115, 115))
  expect_false(cuasmR:::.power_below(150, 150))
  # A driver reporting the pair a hair apart is not a cap.
  expect_false(cuasmR:::.power_below(114.5, 115))
  # Enforced above reference (laptop raised 115 -> 150 by OEM mode).
  expect_false(cuasmR:::.power_below(150, 115))
})

test_that(".power_below returns NA when it cannot tell, never FALSE", {
  # The load-bearing case. "We do not know whether this session was
  # capped" must not be recorded as "this session was not capped" --
  # only the second licenses a comparison against a baseline.
  expect_true(is.na(cuasmR:::.power_below(NULL, 115)))
  expect_true(is.na(cuasmR:::.power_below(50, NULL)))
  expect_true(is.na(cuasmR:::.power_below(NA_real_, 115)))
  expect_true(is.na(cuasmR:::.power_below(50, NA_real_)))
  # nvidia-smi reports "[N/A]" as a string when the field is unsupported.
  expect_true(is.na(cuasmR:::.power_below("[N/A]", 115)))
  expect_true(is.na(cuasmR:::.power_below(50, "[N/A]")))
  expect_true(is.na(cuasmR:::.power_below(50, 0)))
})

test_that("the 115 W fallback regime is NOT invisible", {
  # The exact regression this split exists to prevent. On this machine
  # the OEM performance mode raises the enforced limit to 150 W; when it
  # lapses the limit falls back to the 115 W default. Referenced against
  # `default` alone that reads as perfectly healthy.
  expect_false(cuasmR:::.power_below(115, 115))  # vs default: invisible
  expect_true(cuasmR:::.power_below(115, 150))   # vs max: caught
})

test_that(".as_watts normalises the driver's [N/A] string to NA", {
  # Otherwise the JSONL has a field that is a number on one machine and
  # the string "[N/A]" on another.
  expect_identical(cuasmR:::.as_watts("[N/A]"), NA_real_)
  expect_identical(cuasmR:::.as_watts(NULL), NA_real_)
  expect_identical(cuasmR:::.as_watts(115), 115)
  expect_identical(cuasmR:::.as_watts("115.00"), 115)
})

test_that(".fmt_w never renders NA into a digits-only field", {
  expect_identical(cuasmR:::.fmt_w(50), "50")
  expect_identical(cuasmR:::.fmt_w("[N/A]"), "?")
  expect_identical(cuasmR:::.fmt_w(NULL), "?")
})

test_that("classify_meta shows the power triple only when below ceiling", {
  mk <- function(enforced, default, max) {
    g <- list(clock_sm = 1770, clock_mem = 7001, temp_c = 60,
              power_w = 65, pstate = "P0", throttle = "GpuIdle",
              power_limit_w = enforced, power_limit_default_w = default,
              power_limit_max_w = max)
    g$power_below_default <- cuasmR:::.power_below(enforced, default)
    g$power_below_max     <- cuasmR:::.power_below(enforced, max)
    list(gpu = g, host = list(ac_state = "ac", gpu_mode = "unknown"))
  }

  clamped  <- classify_meta(mk(50, 115, 150),  mk(50, 115, 150))
  fallback <- classify_meta(mk(115, 115, 150), mk(115, 115, 150))
  normal   <- classify_meta(mk(150, 115, 150), mk(150, 115, 150))

  expect_match(clamped$summary,  "POWER-LIMIT=50/115/150W",  fixed = TRUE)
  # The case that used to read as healthy.
  expect_match(fallback$summary, "POWER-LIMIT=115/115/150W", fixed = TRUE)
  # Normal operating point on this machine: enforced == max. Silent.
  expect_false(grepl("POWER-LIMIT", normal$summary, fixed = TRUE))
})

test_that("classify_meta tolerates snapshots with no power-limit fields", {
  # Every pre-#207 caller builds snapshots without these fields; they must
  # keep working and must not gain a spurious POWER-LIMIT marker.
  old <- list(gpu = list(clock_sm = 1770, clock_mem = 7001, temp_c = 60,
                         power_w = 65, pstate = "P0", throttle = "GpuIdle"),
              host = list(ac_state = "ac", gpu_mode = "unknown"))
  res <- classify_meta(old, old)
  expect_true(res$ok)
  expect_false(grepl("POWER-LIMIT", res$summary, fixed = TRUE))
})

test_that("capture_power_policy returns the documented shape", {
  # Deliberately NOT branching on the function's own output the way an
  # earlier version of this test did -- a test whose expectations are
  # chosen by the result it is checking cannot fail. Assert only what is
  # true on EVERY platform: the contract's shape, and that the two
  # source values are the only ones allowed.
  pol <- capture_power_policy()
  expect_true(is.list(pol))
  expect_setequal(names(pol),
                  c("overlay_guid", "overlay_name", "ac_overlay_guid",
                    "dc_overlay_guid", "governing", "source"))
  expect_true(pol$source %in% c("windows-registry", "unavailable"))
})

test_that("capture_power_policy reports 'unavailable' with no powershell", {
  # Force the off-Windows path deterministically, so this assertion runs
  # on the CI runner AND on this machine rather than only where the
  # binary happens to be missing.
  local_mocked_bindings(Sys.which = function(...) "", .package = "base")
  pol <- capture_power_policy(refresh = TRUE)
  expect_identical(pol$source, "unavailable")
  expect_true(is.na(pol$overlay_guid))
  expect_true(is.na(pol$overlay_name))
})

test_that("capture_power_policy rejects a non-GUID token", {
  # Windows PowerShell writes its WARNING stream to stdout, localized,
  # and -ErrorAction does not suppress it. Concatenating stdout blindly
  # produced an observed "warnung: noise<guid>" value that was stored as
  # provenance. A value that is not GUID-shaped is not a reading.
  local_mocked_bindings(
    Sys.which = function(...) "/fake/powershell.exe",
    system2   = function(...) c("WARNUNG: irgendetwas", "not-a-guid|also-not"),
    .package  = "base")
  pol <- capture_power_policy(refresh = TRUE)
  expect_identical(pol$source, "unavailable")
})

test_that("capture_power_policy picks the overlay the power source governs", {
  ac <- "ded574b5-45a0-4f42-8737-46345c09c238"  # best-performance
  dc <- "961cc777-2547-4f9d-8174-7d86181b8a7a"  # best-power-efficiency
  local_mocked_bindings(
    Sys.which = function(...) "/fake/powershell.exe",
    system2   = function(...) paste0(ac, "|", dc),
    .package  = "base")

  on_ac  <- capture_power_policy(ac_state = "ac",      refresh = TRUE)
  on_bat <- capture_power_policy(ac_state = "battery", refresh = TRUE)

  expect_identical(on_ac$overlay_name, "best-performance")
  expect_identical(on_ac$governing, "ac")
  # The real state on this machine on 2026-08-13: the two overlays
  # disagreed, so assuming AC would have recorded the wrong one.
  expect_identical(on_bat$overlay_name, "best-power-efficiency")
  expect_identical(on_bat$governing, "dc")
  # Both raw values are kept either way.
  expect_identical(on_bat$ac_overlay_guid, ac)
  expect_identical(on_bat$dc_overlay_guid, dc)
})

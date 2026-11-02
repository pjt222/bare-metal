# tests/probe/test_conv2d_variance.R
#
# GPU-free tests for the #195 probe harness, scripts/probe/conv2d_variance.R.
#
# #212: the A/B assignment must interleave, not merely balance. With
# block >= n the old loop emitted AAA...BBB, balanced, so its own stopifnot
# passed, while condition was fully confounded with time order. These tests
# pin that the sequence alternates for several (n, block) pairs, that a
# degenerate design is refused, and that lead-in samples are explicit.
#
# #213: the per-sample record asked the snapshot for `power_capped`, a field
# capture_gpu_state() no longer emits, and recorded NA on every row without
# complaint. The schema test below runs the REAL cuasmR::capture_gpu_state()
# with only its nvidia-smi query mocked. So when cuasmR renames a field, this
# suite fails, instead of a campaign quietly recording NA. A hand-built
# fixture would only check the probe against its own assumptions.
#
# Run:
#   Rscript tests/probe/test_conv2d_variance.R

library(testthat)

# main() is guarded by `if (sys.nframe() == 0L)`, so sourcing runs no bench.
# Relative candidates only, resolved from the repo root, which is where
# scripts/audit/run_r_tests.R runs its children (#173: no hardcoded paths).
.candidates <- c(
  "scripts/probe/conv2d_variance.R",
  file.path(getwd(), "scripts", "probe", "conv2d_variance.R"))
.src <- NULL
for (.p in .candidates) if (file.exists(.p)) { .src <- .p; break }
if (is.null(.src)) {
  stop("can't find scripts/probe/conv2d_variance.R -- run this from the repo root")
}
suppressMessages(source(.src))

# A raw nvidia-smi answer in the shape cuasmR's internal query returns: one
# value per field it asks for, numerics already parsed. Field names come from
# cuasmR itself, so a change to its query list is picked up here as well.
.fake_smi <- function(enforced = 150, default = 115, max = 150,
                      throttle_hex = "0x0000000000000001") {
  fields <- get(".NVIDIA_SMI_FIELDS", envir = asNamespace("cuasmR"))
  vals <- lapply(fields, function(f) switch(f,
    "pstate"                         = "P8",
    "clocks_throttle_reasons.active" = throttle_hex,
    "enforced.power.limit"           = enforced,
    "power.default_limit"            = default,
    "power.max_limit"                = max,
    1))
  setNames(vals, fields)
}

.real_snapshot <- function(...) {
  raw <- .fake_smi(...)
  testthat::with_mocked_bindings(
    cuasmR::capture_gpu_state(),
    .nvidia_smi_query = function(...) raw,
    .package = "cuasmR")
}

# --- #212: assignment ---------------------------------------------------------

test_that("the A/B sequence is balanced AND interleaved for valid designs", {
  for (d in list(c(2, 1), c(3, 1), c(3, 2), c(4, 3), c(15, 1), c(15, 4),
                 c(15, 5), c(15, 14))) {
    n <- as.integer(d[1]); block <- as.integer(d[2])
    s <- assign_conditions(n, block)
    label <- sprintf("n=%d block=%d: %s", n, block, paste(s, collapse = ""))
    expect_equal(sum(s == "A"), n, label = label)
    expect_equal(sum(s == "B"), n, label = label)
    # Interleaved: more than the two runs of AAA...BBB.
    expect_gt(length(rle(s)$lengths), 2L, label = label)
    # No run longer than the block size.
    expect_lte(max(rle(s)$lengths), block, label = label)
  }
})

test_that("--block 1 alternates strictly", {
  expect_identical(paste(assign_conditions(15L, 1L), collapse = ""),
                   strrep("AB", 15L))
})

test_that("a degenerate or invalid design is refused, not run", {
  expect_error(assign_conditions(3L, 5L), "confounds condition with time order")
  expect_error(assign_conditions(3L, 3L), "confounds condition with time order")
  expect_error(assign_conditions(15L, 15L), "confounds condition with time order")
  expect_error(assign_conditions(1L, 1L), "--n must be an integer >= 2")
  expect_error(assign_conditions(NA_integer_, 1L), "--n must be an integer >= 2")
  expect_error(assign_conditions(15L, 0L), "--block must be an integer >= 1")
  expect_error(assign_conditions(15L, NA_integer_), "--block must be an integer >= 1")
})

test_that("the default design is --block 1 with no burn-in", {
  a <- parse_args(character(0))
  expect_identical(a$block, 1L)
  expect_identical(a$burnin, 0L)
})

test_that("burn-in samples precede the design and are marked W", {
  s <- with_burnin(assign_conditions(3L, 1L), 2L)
  expect_identical(paste(s, collapse = ""), "WWABABAB")
  expect_identical(with_burnin(c("A", "B"), 0L), c("A", "B"))
  expect_error(with_burnin("A", -1L), "--burnin must be an integer >= 0")
  expect_error(with_burnin("A", NA_integer_), "--burnin must be an integer >= 0")
})

# --- #213: per-sample record ----------------------------------------------------

test_that("every recorded field exists in a real capture_gpu_state() snapshot", {
  snap <- .real_snapshot()
  f <- gpu_fields(snap, "pre_")
  expect_true(f$pre_snapshot)
  # No recorded column may be NA when the driver reported everything: that
  # was the #213 symptom (power_capped NA on 36 of 36 rows).
  power_and_load <- c("pre_power_limit_w", "pre_power_limit_default_w",
                      "pre_power_limit_max_w", "pre_power_below_default",
                      "pre_power_below_max")
  for (nm in power_and_load) expect_false(is.na(f[[nm]]), label = nm)
  expect_equal(f$pre_power_limit_w, 150)
  expect_equal(f$pre_power_limit_default_w, 115)
  expect_false(f$pre_power_below_max)
  expect_false(f$pre_power_below_default)
})

test_that("both power references are recorded, and a cap shows in each", {
  f <- gpu_fields(.real_snapshot(enforced = 115), "")
  expect_true(f$power_below_max)       # 115 < 150 - 1
  expect_false(f$power_below_default)  # 115 is the default
  f50 <- gpu_fields(.real_snapshot(enforced = 50), "")
  expect_true(f50$power_below_max)
  expect_true(f50$power_below_default)
})

test_that("a field missing from the snapshot stops the run (schema drift)", {
  snap <- .real_snapshot()
  snap$gpu$power_below_max <- NULL
  expect_error(gpu_fields(snap, "pre_"), "no field\\(s\\) power_below_max")
})

test_that("no snapshot is recorded as such, with the same columns", {
  snap <- .real_snapshot()
  with_snap <- gpu_fields(snap, "post_")
  without <- gpu_fields(NULL, "post_")
  expect_identical(names(without), names(with_snap))
  expect_false(without$post_snapshot)
  expect_true(all(is.na(unlist(without[setdiff(names(without), "post_snapshot")]))))
})

test_that("loadavg is recorded when the host provides it", {
  snap <- .real_snapshot()
  snap$host$loadavg <- list(load_1m = 2.5, load_5m = 3, load_15m = 3.5)
  f <- gpu_fields(snap, "")
  expect_equal(c(f$load_1m, f$load_5m, f$load_15m), c(2.5, 3, 3.5))
  snap$host$loadavg <- NULL
  expect_true(is.na(gpu_fields(snap, "")$load_1m))
})

test_that("an unreadable throttle mask is NA, not the string 'NA'", {
  snap <- .real_snapshot()
  snap$gpu$throttle <- NA_character_
  expect_true(is.na(gpu_fields(snap, "")$throttle))
  snap$gpu$throttle <- character(0)
  expect_identical(gpu_fields(snap, "")$throttle, "")
  expect_identical(gpu_fields(.real_snapshot(), "")$throttle, "GpuIdle")
})

# --- #224 review, H1: never pool two campaigns into one file -----------------

test_that("an existing --jsonl is refused unless --append is given", {
  existing <- tempfile(fileext = ".jsonl")
  writeLines('{"run_id":"old"}', existing)
  on.exit(unlink(existing), add = TRUE)
  expect_error(check_output_path(existing, FALSE), "already exists")
  expect_silent(check_output_path(existing, TRUE))
  expect_silent(check_output_path(tempfile(fileext = ".jsonl"), FALSE))
  expect_false(parse_args(character(0))$append)
  expect_true(parse_args("--append")$append)
})

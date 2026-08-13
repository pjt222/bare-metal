# decode_throttle: the throttle mask is 64-bit (#208).
#
# The old implementation ran the whole mask through strtoi(base = 16L),
# which is int32. Any mask with a bit at or above 2^31 returned NA, and the
# NA branch returned character(0) -- the SAME value that means "no
# throttle". A throttled run therefore decoded as a clean one, passed
# classify_meta, and was compared against a baseline. The failure direction
# is what makes it serious: it launders unfair runs as fair.

test_that("decode_throttle reads the documented Ampere reasons", {
  expect_identical(decode_throttle("0x0000000000000001"), "GpuIdle")
  expect_identical(decode_throttle("0x0000000000000004"), "SwPowerCap")
  expect_identical(decode_throttle("0x0000000000000024"),
                   c("SwPowerCap", "SwThermalSlowdown"))
  expect_identical(decode_throttle("0x0000000000000100"), "DisplayClocksSetting")
})

test_that("decode_throttle returns character(0) only for a real zero mask", {
  expect_identical(decode_throttle("0x0000000000000000"), character(0))
  expect_identical(decode_throttle("0x0"), character(0))
})

test_that("a high-bit mask no longer decodes as 'no throttle'", {
  # THE REGRESSION. 0x80000004 exceeds .Machine$integer.max, so the old
  # strtoi returned NA and the function reported character(0) -- "clean" --
  # while SwPowerCap (0x4) was set in the mask.
  expect_identical(decode_throttle("0x0000000080000004"), "SwPowerCap")
  expect_identical(decode_throttle("0x8000000000000004"), "SwPowerCap")
  expect_identical(decode_throttle("0x0000000100000001"), "GpuIdle")
  # Sanity: the value that used to break strtoi is genuinely out of int32.
  expect_true(is.na(suppressWarnings(strtoi("0000000080000004", 16L))))
})

test_that("an unparseable mask returns NA, not 'no throttle'", {
  # These two must never be the same answer. character(0) says "the GPU was
  # not throttled", which licenses a comparison against a baseline; NA says
  # "we could not tell", which must not.
  expect_identical(decode_throttle("0xnonsense"), NA_character_)
  expect_identical(decode_throttle("0x00000000000000000"), NA_character_)
  expect_false(identical(decode_throttle("0xnonsense"), character(0)))
})

test_that("an unparseable mask makes classify_meta reject the sample", {
  # The fail-safe direction, end to end. setdiff() surfaces the NA as an
  # unrecognised reason, so the run is skipped rather than measured.
  s <- list(gpu = list(clock_sm = 1770, clock_mem = 7001, temp_c = 60,
                       power_w = 65, pstate = "P0",
                       throttle = decode_throttle("0xnonsense")),
            host = list(ac_state = "ac", gpu_mode = "unknown"))
  cls <- classify_meta(s, s)
  expect_false(isTRUE(cls$ok))
  expect_true(any(grepl("throttle", cls$reasons)))
})

test_that("decode_throttle tolerates absent input", {
  expect_identical(decode_throttle(NULL), character(0))
  expect_identical(decode_throttle(""), character(0))
  expect_identical(decode_throttle(NA_character_), character(0))
})

test_that("nvidia-smi's [N/A] is 'no data', the same as NULL -- not a parse failure", {
  # Review finding: the first version gave OPPOSITE answers to the two ways
  # "field unavailable" arrives -- NA_character_ -> character(0) ("clean"),
  # but the literal "[N/A]" -> NA_character_ (reject). On a driver that does
  # not report throttle reasons that second path rejects 100% of samples,
  # a failure mode #208 never asked for. Both are "the driver told us
  # nothing" and must agree.
  expect_identical(decode_throttle("[N/A]"), character(0))
  expect_identical(decode_throttle("N/A"), character(0))
  expect_identical(decode_throttle(" [N/A] "), character(0))
  expect_identical(decode_throttle("[n/a]"), character(0))
  # Genuinely malformed input is still a parse failure, not "clean".
  expect_identical(decode_throttle("0xnonsense"), NA_character_)
})

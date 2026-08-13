# hex64_to_bytes converts a "0x...." or "...." hex string to 8 little-endian raw
# bytes. Short and odd-length inputs used to survive only by three cancelling
# coercion accidents (space padding -> NA pairs -> as.raw(NA) == 00, with a
# warning); expect_no_warning() is what pins the explicit padding. See #170.

test_that("hex64_to_bytes zero-pads a short hex string without warning", {
    expect_no_warning(bytes <- cuasmR:::hex64_to_bytes("1a2b"))
    expect_identical(bytes, as.raw(c(0x2b, 0x1a, 0, 0, 0, 0, 0, 0)))
    # The "0x" prefix and uppercase are both accepted.
    expect_identical(cuasmR:::hex64_to_bytes("0X1A2B"), bytes)
})

test_that("hex64_to_bytes handles an odd-length hex string without warning", {
    expect_no_warning(bytes <- cuasmR:::hex64_to_bytes("abc"))
    expect_identical(bytes, as.raw(c(0xbc, 0x0a, 0, 0, 0, 0, 0, 0)))
})

test_that("hex64_to_bytes round-trips a full-width instruction word", {
    # The FADD word from kernels/tutorial/vector_add.sm_86.cubin.
    expect_no_warning(bytes <- cuasmR:::hex64_to_bytes("0x0000000304097221"))
    expect_identical(bytes, as.raw(c(0x21, 0x72, 0x09, 0x04, 0x03, 0, 0, 0)))
})

test_that("hex64_to_bytes rejects an over-long string", {
    expect_error(cuasmR:::hex64_to_bytes("1234567890abcdef0"), "too long")
})

test_that("hex64_to_bytes rejects a non-string instead of coercing it", {
    # 0x1337 is valid R for the double 4919, not a string. Without a type gate
    # tolower() stringifies it to "4919", which is all hex digits and so passes
    # the content check -- writing a different value than the caller meant.
    expect_error(cuasmR:::hex64_to_bytes(0x1337), "length-1 character")
    expect_error(cuasmR:::hex64_to_bytes(4919L), "length-1 character")
    expect_error(cuasmR:::hex64_to_bytes(c("1a2b", "abc")), "length 2")
    expect_error(cuasmR:::hex64_to_bytes(character(0)), "length 0")
    expect_error(cuasmR:::hex64_to_bytes(NA_character_), "is NA")
})

test_that("hex64_to_bytes rejects malformed input instead of zeroing it", {
    # sprintf("0x%016x", NA) yields exactly this token: 16 chars after the "0x"
    # strip, so it clears the length check. Before #170 it wrote eight 00 bytes
    # into a live instruction word and only warned. See #197.
    expect_error(cuasmR:::hex64_to_bytes("0x              NA"), "not hex")
    expect_error(cuasmR:::hex64_to_bytes(""), "not hex")
    expect_error(cuasmR:::hex64_to_bytes("0xzz"), "not hex")
})

# ---- hex64 bit/nibble editing (#198, #208) ---------------------------------
#
# The rule these enforce: never hand a 64-bit word to strtoi(). It returns
# int32, so every 64-bit word and every 32-bit half with a leading digit
# >= 8 becomes NA, and sprintf("%016x", NA) turns that into the literal
# "0x              NA" -- a plausible-width token that is written out as
# eight zero bytes.

test_that("hex64_bit_set edits the bit the old strtoi path could not reach", {
    # The real oxide FADD control word. strtoi() on all 16 digits returns NA,
    # so run_oxide.sh produced "0x              NA" and would have zeroed the
    # control word on regeneration instead of setting bit 22 (#198).
    expect_identical(hex64_bit_set("0x004fe20000000000", 22L),
                     "0x004fe20000400000")
    # ...which is exactly the committed vecadd_oxide.fmul.cubin's word.
    expect_true(hex64_bit_get("0x004fe20000400000", 22L))
    expect_false(hex64_bit_get("0x004fe20000000000", 22L))
})

test_that("hex64_bit_get/set reach both ends of the 64-bit range", {
    expect_identical(hex64_bit_set("0x0000000000000000", 0L),  "0x0000000000000001")
    expect_identical(hex64_bit_set("0x0000000000000000", 63L), "0x8000000000000000")
    expect_true(hex64_bit_get("0x8000000000000000", 63L))
    expect_true(hex64_bit_get("0x0000000000000001", 0L))
    # Bit 31/32 -- the int32 boundary that started all of this.
    expect_identical(hex64_bit_set("0x0000000000000000", 31L), "0x0000000080000000")
    expect_identical(hex64_bit_set("0x0000000000000000", 32L), "0x0000000100000000")
    expect_true(hex64_bit_get("0x0000000080000000", 31L))
})

test_that("hex64_bit_set clears, and is its own inverse", {
    w <- "0x004fe20000400000"
    expect_identical(hex64_bit_set(w, 22L, FALSE), "0x004fe20000000000")
    expect_identical(hex64_bit_set(hex64_bit_set(w, 5L, TRUE), 5L, FALSE), w)
    # Setting a bit that is already set changes nothing.
    expect_identical(hex64_bit_set(w, 22L, TRUE), w)
})

test_that("hex64_nibble_get/set handle the SASS stall field (bits 40..43)", {
    # Nibble 10 is hex digit 6 counted from the left (16 - 10). Every
    # observed sm_86 control word starts 0x001f/0x004f/0x004e.
    expect_identical(hex64_nibble_get("0x001f8000fc0007f0", 10L), 0L)
    expect_identical(hex64_nibble_set("0x001f8000fc0007f0", 10L, 4L),
                     "0x001f8400fc0007f0")
    # Cross-check against the arithmetic the old ctrl_to_stall used, on a
    # word where that path still worked (leading digit < 8).
    expect_identical(
        hex64_nibble_get("0x001f8000fc0007f0", 10L),
        bitwAnd(bitwShiftR(strtoi("001f8000", 16L), 8), 0xF))
    # A leading digit >= 8 broke the old top-half strtoi; nibble access does not.
    expect_identical(hex64_nibble_get("0x804fe20000000000", 10L), 2L)
    expect_identical(hex64_nibble_set("0x804fe20000000000", 10L, 4L),
                     "0x804fe40000000000")
})

test_that("hex64 helpers validate input instead of coercing it", {
    expect_error(hex64_bit_get(0x1337, 0L), "length-1 character")
    expect_error(hex64_bit_get(NA_character_, 0L), "is NA")
    expect_error(hex64_bit_get("0xzz", 0L), "not hex")
    expect_error(hex64_bit_get("0x00000000000000000", 0L), "too long")
    expect_error(hex64_bit_get("0x0", 64L), "0\\.\\.63")
    expect_error(hex64_bit_get("0x0", -1L), "0\\.\\.63")
    expect_error(hex64_bit_set("0x0", 0L, NA), "TRUE/FALSE")
    expect_error(hex64_nibble_set("0x0", 0L, 16L), "0\\.\\.15")
    expect_error(hex64_nibble_set("0x0", 0L, -1L), "0\\.\\.15")
})

test_that("hex64 accepts input with or without the 0x prefix", {
    expect_identical(hex64_bit_set("004fe20000000000", 22L), "0x004fe20000400000")
    expect_identical(hex64_bit_set("0X004FE20000000000", 22L), "0x004fe20000400000")
    # Short input is zero-padded, not misaligned.
    expect_identical(hex64_bit_set("1", 4L), "0x0000000000000011")
})

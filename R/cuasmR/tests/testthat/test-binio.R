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

test_that("hex64_to_bytes rejects malformed input instead of zeroing it", {
    # sprintf("0x%016x", NA) yields exactly this token: 16 chars after the "0x"
    # strip, so it clears the length check. Before #170 it wrote eight 00 bytes
    # into a live instruction word and only warned. See #197.
    expect_error(cuasmR:::hex64_to_bytes("0x              NA"), "not hex")
    expect_error(cuasmR:::hex64_to_bytes(""), "not hex")
    expect_error(cuasmR:::hex64_to_bytes("0xzz"), "not hex")
})

test_that("byte-identical roundtrip on phase1 vector_add", {
    skip_if_not(file.exists(test_path("../../../../kernels/tutorial/vector_add.sm_86.cubin")),
                "cubin not built")
    skip_if(Sys.which("nvdisasm") == "", "nvdisasm not on PATH")
    p <- test_path("../../../../kernels/tutorial/vector_add.sm_86.cubin")
    expect_true(cuasm_roundtrip_check(p))
})

test_that("cubin layout decode handles CUDA 12.x and 13.x", {
    skip_if_not(file.exists(test_path("../../../../kernels/tutorial/vector_add.sm_86.cubin")),
                "cubin not built")
    skip_if(Sys.which("nvdisasm") == "", "nvdisasm not on PATH")
    obj <- cuasm_read(test_path("../../../../kernels/tutorial/vector_add.sm_86.cubin"))
    expect_true(obj$arch$sm_version %in% c(75, 80, 86, 87, 89, 90))
    expect_true(obj$arch$layout %in% c("cuda12", "cuda13"))
})

test_that("cuasm_set patches a single 16-byte slot only", {
    skip_if_not(file.exists(test_path("../../../../kernels/tutorial/vector_add.sm_86.cubin")),
                "cubin not built")
    skip_if(Sys.which("nvdisasm") == "", "nvdisasm not on PATH")
    p <- test_path("../../../../kernels/tutorial/vector_add.sm_86.cubin")
    obj <- cuasm_read(p)

    # Pick the FADD slot by mnemonic
    rows <- which(grepl("^FADD", obj$insns$text))
    skip_if(length(rows) == 0, "no FADD found in vector_add")
    row <- rows[1]

    orig_instr <- obj$insns$instr_hex[row]
    # Toggle a single non-opcode bit (bit 12 in the instr word) by flipping the
    # hex digit directly. strtoi() returns int32, so handing it the whole 64-bit
    # word gives NA, sprintf("%016x", NA) gives a 16-char token, and the write
    # zeroes the instruction word instead of toggling a bit. See #197.
    instr_body    <- sub("^0x", "", orig_instr)
    bit12_digit   <- 13L   # bit 12 is the low bit of the 4th hex digit from the right
    orig_digit    <- strtoi(substring(instr_body, bit12_digit, bit12_digit), 16L)
    new_hex <- paste0("0x",
                      substring(instr_body, 1L, bit12_digit - 1L),
                      sprintf("%x", bitwXor(orig_digit, 1L)),
                      substring(instr_body, bit12_digit + 1L))
    obj <- cuasm_set(obj, kernel = obj$insns$kernel[row],
                     slot = obj$insns$slot[row], instr_hex = new_hex)

    out <- tempfile(fileext = ".cubin")
    on.exit(unlink(out), add = TRUE)
    cuasm_write(obj, out)

    a <- readBin(p,  "raw", n = file.info(p)$size)
    b <- readBin(out, "raw", n = file.info(out)$size)
    n_diff <- sum(a != b)
    # Exactly one byte, and exactly bit 12 within it. The old bound
    # (0 < n_diff <= 8) passed while the write zeroed five bytes; a count alone
    # is still not enough, since flipping digit 12, 13 or 14 each changes
    # exactly one byte and only 13 is bit 12.
    expect_equal(n_diff, 1L)
    changed_byte <- which(a != b)
    expect_equal(bitwXor(as.integer(a[changed_byte]),
                         as.integer(b[changed_byte])), 0x10)
})

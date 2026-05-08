test_that("byte-identical roundtrip on phase1 vector_add", {
    skip_if_not(file.exists(test_path("../../../../phase1/vector_add.sm_86.cubin")),
                "cubin not built")
    p <- test_path("../../../../phase1/vector_add.sm_86.cubin")
    expect_true(cuasm_roundtrip_check(p))
})

test_that("cubin layout decode handles CUDA 12.x and 13.x", {
    skip_if_not(file.exists(test_path("../../../../phase1/vector_add.sm_86.cubin")),
                "cubin not built")
    obj <- cuasm_read(test_path("../../../../phase1/vector_add.sm_86.cubin"))
    expect_true(obj$arch$sm_version %in% c(75, 80, 86, 87, 89, 90))
    expect_true(obj$arch$layout %in% c("cuda12", "cuda13"))
})

test_that("cuasm_set patches a single 16-byte slot only", {
    skip_if_not(file.exists(test_path("../../../../phase1/vector_add.sm_86.cubin")),
                "cubin not built")
    p <- test_path("../../../../phase1/vector_add.sm_86.cubin")
    obj <- cuasm_read(p)

    # Pick the FADD slot by mnemonic
    rows <- which(grepl("^FADD", obj$insns$text))
    skip_if(length(rows) == 0, "no FADD found in vector_add")
    row <- rows[1]

    orig_instr <- obj$insns$instr_hex[row]
    # Toggle a single non-opcode bit (bit 12 in the instr word)
    new_int <- bitwXor(strtoi(sub("0x", "", orig_instr), 16L), 0x1000)
    new_hex <- sprintf("0x%016x", new_int)
    obj <- cuasm_set(obj, kernel = obj$insns$kernel[row],
                     slot = obj$insns$slot[row], instr_hex = new_hex)

    out <- tempfile(fileext = ".cubin")
    on.exit(unlink(out), add = TRUE)
    cuasm_write(obj, out)

    a <- readBin(p,  "raw", n = file.info(p)$size)
    b <- readBin(out, "raw", n = file.info(out)$size)
    n_diff <- sum(a != b)
    # At most 8 bytes should differ (one 64-bit word's worth of bits).
    expect_true(n_diff > 0)
    expect_true(n_diff <= 8)
})

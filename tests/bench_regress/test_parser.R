# tests/bench_regress/test_parser.R
#
# Unit tests for the bench_regress.R output parser. Feeds canned bench
# stdout strings and asserts the parser picks the right row + column.
#
# Run with:
#   Rscript tests/bench_regress/test_parser.R
#
# Or via testthat::test_file("tests/bench_regress/test_parser.R").

library(testthat)

# Source the parser helpers from the script. The script's main() is
# guarded by `if (sys.nframe() == 0L) main()` so sourcing alone won't
# trigger any benchmark runs.
.candidates <- c(
  "scripts/bench/bench_regress.R",
  file.path(getwd(), "scripts", "bench", "bench_regress.R"),
  "/mnt/d/dev/p/bare-metal/scripts/bench/bench_regress.R"
)
.src <- NULL
for (.p in .candidates) if (file.exists(.p)) { .src <- .p; break }
if (is.null(.src)) stop("can't find bench_regress.R")
source(.src)

# ---- HGEMM bench: multi-kernel rows, pick the 16warp variant ---------
HGEMM_OUT <- c(
  "=== HGEMM Benchmark — Tensor Cores (sm_86) ===",
  "Matrix: C[2048x2048] = A[2048x2048] * B[2048x2048]  (FP16 in, FP32 out)",
  "",
  "Performance (avg of 50 runs, 5 warmup):",
  "  hgemm_wmma (naive 32x32)              2.437 ms    7048.54 GFLOPS",
  "  hgemm_tiled (128x128 smem epi)        1.383 ms   12421.31 GFLOPS",
  "  hgemm_direct (128x128 no epi)         1.291 ms   13309.12 GFLOPS",
  "  hgemm_16warp (128x128 2blk/SM)        0.536 ms   32022.48 GFLOPS",
  "  hgemm_16warp_epi (2blk+epi)           0.946 ms   18157.16 GFLOPS"
)

test_that("hgemm: legacy first-line parsing picks naive (broken default)", {
  line <- .pick_line(HGEMM_OUT)
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 2.437)
  expect_equal(parsed$throughput, 7048.54)
  expect_equal(parsed$unit, "GFLOPS")
})

test_that("hgemm: match='hgemm_16warp (' picks the right variant", {
  line <- .pick_line(HGEMM_OUT, match_str = "hgemm_16warp (128x128 2blk/SM)")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 0.536)
  expect_equal(parsed$throughput, 32022.48)
  expect_equal(parsed$unit, "GFLOPS")
})

test_that("hgemm: match='hgemm_16warp_epi' picks the epi variant", {
  line <- .pick_line(HGEMM_OUT, match_str = "hgemm_16warp_epi")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 0.946)
  expect_equal(parsed$throughput, 18157.16)
})

test_that("hgemm: unknown match returns NULL", {
  line <- .pick_line(HGEMM_OUT, match_str = "does_not_exist")
  expect_null(line)
})

# ---- Sparse INT8: line has multiple numbers; need value_label --------
SPARSE_OUT <- c(
  "=== sparse INT8 GEMM (sm_86) ===",
  "  igemm_sparse_tiled (INT8)      PASS  (max_abs=0.00e+00  max_rel=0.00e+00)",
  "  igemm_sparse_tiled                     0.545 ms     15762 eff GFLOPS     31524 dense-equiv GFLOPS"
)

test_that("sparse: default parser is ambiguous on multi-column lines", {
  # Line is 'X ms  N1 eff GFLOPS  N2 dense-equiv GFLOPS' — there is no
  # bare 'N GFLOPS' substring (always a prefix word), so the legacy
  # `([0-9.]+)\s*(GFLOPS|TOPS)` regex finds nothing and throughput is
  # NULL. value_label is required to disambiguate.
  line <- .pick_line(SPARSE_OUT, match_str = "igemm_sparse_tiled")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 0.545)
  expect_null(parsed$throughput)
})

test_that("sparse: value_label='dense-equiv GFLOPS' picks the dense column", {
  line <- .pick_line(SPARSE_OUT, match_str = "igemm_sparse_tiled")
  parsed <- .parse_line(line, value_label = "dense-equiv GFLOPS")
  expect_equal(parsed$ms, 0.545)
  expect_equal(parsed$throughput, 31524)
  expect_equal(parsed$unit, "GFLOPS")
})

test_that("sparse: value_label='eff GFLOPS' picks the effective column", {
  line <- .pick_line(SPARSE_OUT, match_str = "igemm_sparse_tiled")
  parsed <- .parse_line(line, value_label = "eff GFLOPS")
  expect_equal(parsed$throughput, 15762)
})

# ---- conv2d implicit_gemm: section headers bracket multiple shapes ---
CONV_OUT <- c(
  "=== Implicit GEMM vs Explicit im2col + GEMM ===",
  "",
  "--- SD 64x64  Cin=Cout=320  (baseline, large col buffer) ---",
  "  Explicit (im2col+GEMM): 1.206 ms  -> 6260 GFLOPS",
  "  Implicit (single kern): 0.931 ms  -> 8112 GFLOPS  (1.30x speedup)",
  "",
  "--- SD 32x32  Cin=Cout=640  (smaller spatial, larger channels) ---",
  "  Explicit (im2col+GEMM): 1.715 ms  -> 4402 GFLOPS",
  "  Implicit (single kern): 1.563 ms  -> 4831 GFLOPS  (1.10x speedup)",
  "",
  "--- SD 128x128 Cin=Cout=160 (large spatial, small col buffer penalty) ---",
  "  Explicit (im2col+GEMM): 2.069 ms  -> 3649 GFLOPS",
  "  Implicit (single kern): 1.092 ms  -> 6916 GFLOPS  (1.90x speedup)"
)

test_that("conv2d: section='SD 64' restricts to first SD-64 block", {
  line <- .pick_line(CONV_OUT,
                     match_str = "Implicit (single kern)",
                     section_str = "SD 64")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 0.931)
  expect_equal(parsed$throughput, 8112)
})

test_that("conv2d: section='SD 32' restricts to the SD-32 block", {
  line <- .pick_line(CONV_OUT,
                     match_str = "Implicit (single kern)",
                     section_str = "SD 32")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 1.563)
  expect_equal(parsed$throughput, 4831)
})

test_that("conv2d: section='SD 128' restricts to the SD-128 block", {
  line <- .pick_line(CONV_OUT,
                     match_str = "Implicit (single kern)",
                     section_str = "SD 128")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 1.092)
  expect_equal(parsed$throughput, 6916)
})

test_that("conv2d: no section returns first matching line across all sections", {
  line <- .pick_line(CONV_OUT, match_str = "Implicit (single kern)")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 0.931)  # first match: SD 64
})

# ---- TOPS unit handling ----------------------------------------------
IGEMM_OUT <- c(
  "=== IGEMM Benchmark ===",
  "  igemm_wmma  (naive)                   1.793 ms    9582.16 TOPS",
  "  igemm_cpasync (LDGSTS dbuf)           0.838 ms   20496.51 TOPS"
)

test_that("igemm: TOPS unit is preserved", {
  line <- .pick_line(IGEMM_OUT, match_str = "igemm_cpasync")
  parsed <- .parse_line(line)
  expect_equal(parsed$ms, 0.838)
  expect_equal(parsed$throughput, 20496.51)
  expect_equal(parsed$unit, "TOPS")
})

# ---- Edge cases ------------------------------------------------------
test_that("empty output: pick_line returns NULL", {
  expect_null(.pick_line(character(0)))
  expect_null(.pick_line(c("no metrics here", "still nothing")))
})

test_that("section header detection: --- and === both work", {
  expect_true(.is_section_header("--- SD 64x64 ---"))
  expect_true(.is_section_header("=== Header ==="))
  expect_true(.is_section_header("  --- indented ---"))
  expect_false(.is_section_header("  hgemm_16warp 0.5 ms 30000 GFLOPS"))
  expect_false(.is_section_header(""))
})

cat("\nAll bench_regress parser tests passed.\n")

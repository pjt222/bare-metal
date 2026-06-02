# Unit test for the JSONL store (issue #134, Phase 4). GPU-free.

test_that("append_jsonl_row + read_jsonl round-trip", {
  p <- tempfile(fileext = ".jsonl")
  on.exit(unlink(p), add = TRUE)
  append_jsonl_row(p, list(cell_id = "a", clock_target_mhz = 1605L, throughput = 50000))
  append_jsonl_row(p, list(cell_id = "b", clock_target_mhz = NA, throughput = 36000))

  r <- read_jsonl(p, simplify = FALSE)
  expect_equal(r$n_total, 2L)
  expect_equal(r$n_bad, 0L)
  expect_length(r$rows, 2L)
  expect_equal(r$rows[[1]]$cell_id, "a")
  expect_equal(r$rows[[1]]$clock_target_mhz, 1605L)
  # NA serialised as JSON null -> parses back as NULL
  expect_null(r$rows[[2]]$clock_target_mhz)
})

test_that("read_jsonl drops a truncated final line and counts it", {
  p <- tempfile(fileext = ".jsonl")
  on.exit(unlink(p), add = TRUE)
  append_jsonl_row(p, list(cell_id = "a", throughput = 1))
  append_jsonl_row(p, list(cell_id = "b", throughput = 2))
  cat('{"cell_id":"c","throughput":', file = p, append = TRUE)  # truncated tail

  r <- read_jsonl(p, simplify = TRUE)
  expect_equal(r$n_total, 3L)
  expect_equal(r$n_bad, 1L)
  expect_length(r$rows, 2L)
})

test_that("read_jsonl on a missing or empty file yields an empty result", {
  expect_equal(read_jsonl(tempfile())$n_total, 0L)
  expect_length(read_jsonl(NULL)$rows, 0L)
  p <- tempfile(); file.create(p); on.exit(unlink(p), add = TRUE)
  expect_equal(read_jsonl(p)$n_total, 0L)
})

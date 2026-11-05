# Differential / characterization test for parse_throughput (issue #134,
# Phase 2b). Proves the unified parser reproduces, bit-for-bit on real
# captured bench output, the three parser variants it replaces:
#
#   * bench_regress.R  .pick_line + .parse_line   (pick = first)
#   * grid_measure.R   parse_bench_line(out, match, value_label, section)
#   * rebaseline/clock_lock_sweep  parse_bench_line(out, match, label)
#
# The three originals are inlined VERBATIM below as oracles. Fixtures in
# tests/testthat/fixtures/ are raw stdout captured from the actual bench
# executables (hgemm, igemm_sparse, igemm_cpasync, flash, conv2d).

# ---- Oracle 1: bench_regress.R (verbatim) -----------------------------
.o_is_section_header <- function(line) {
  s <- trimws(line)
  grepl("^(===|---)", s, perl = TRUE)
}
.o_filter_section <- function(lines, section_match) {
  if (is.null(section_match) || !nzchar(section_match)) return(lines)
  in_section <- FALSE
  keep <- logical(length(lines))
  for (i in seq_along(lines)) {
    is_hdr <- .o_is_section_header(lines[[i]])
    if (is_hdr) {
      in_section <- grepl(section_match, lines[[i]], fixed = TRUE)
      next
    }
    keep[[i]] <- in_section
  }
  lines[keep]
}
.o_parse_line <- function(line, value_label = NULL) {
  out <- list()
  m_ms <- regmatches(line, regexec("([0-9.]+)\\s*ms", line, perl = TRUE))[[1]]
  if (length(m_ms) >= 2) out$ms <- as.numeric(m_ms[2])
  if (!is.null(value_label) && nzchar(value_label)) {
    pat <- sprintf("([0-9][0-9,.]*)\\s*%s",
                   gsub("([][}{()|.+*?^$\\\\])", "\\\\\\1", value_label, perl = TRUE))
    m <- regmatches(line, regexec(pat, line, perl = TRUE, ignore.case = TRUE))[[1]]
    if (length(m) >= 2) {
      out$throughput <- as.numeric(gsub(",", "", m[2], fixed = TRUE))
      u <- regmatches(value_label, regexec("(GFLOPS|TOPS)\\b", value_label,
                                           perl = TRUE, ignore.case = TRUE))[[1]]
      out$unit <- if (length(u) >= 2) toupper(u[2]) else "GFLOPS"
    }
  } else {
    m_tp <- regmatches(line, regexec("([0-9][0-9,.]*)\\s*(GFLOPS|TOPS)",
                                     line, perl = TRUE, ignore.case = TRUE))[[1]]
    if (length(m_tp) >= 3) {
      out$throughput <- as.numeric(gsub(",", "", m_tp[2], fixed = TRUE))
      out$unit <- toupper(m_tp[3])
    }
  }
  out
}
.o_pick_line <- function(output_lines, match_str = NULL, section_str = NULL) {
  candidates <- .o_filter_section(output_lines, section_str)
  has_metrics <- grepl("ms", candidates) &
                 grepl("GFLOPS|TOPS", candidates, ignore.case = TRUE)
  candidates <- candidates[has_metrics]
  if (!length(candidates)) return(NULL)
  if (is.null(match_str) || !nzchar(match_str)) return(candidates[[1]])
  hits <- candidates[grepl(match_str, candidates, fixed = TRUE)]
  if (length(hits)) hits[[1]] else NULL
}
oracle_bench_regress <- function(out, match = NULL, section = NULL, value_label = NULL) {
  picked <- .o_pick_line(out, match_str = match, section_str = section)
  if (is.null(picked)) picked <- paste(out, collapse = "\n")
  p <- .o_parse_line(picked, value_label = value_label)
  list(ms = if (is.null(p$ms)) NA_real_ else p$ms,
       throughput = if (is.null(p$throughput)) NA_real_ else p$throughput,
       unit = if (is.null(p$unit)) NA_character_ else p$unit)
}

# ---- Oracle 2: grid_measure.R (verbatim) ------------------------------
oracle_grid <- function(out, match, value_label, section) {
  lines <- out
  if (!is.null(section) && nzchar(section)) {
    in_sec <- FALSE
    keep <- logical(length(lines))
    for (i in seq_along(lines)) {
      s <- trimws(lines[[i]])
      is_hdr <- grepl("^(===|---)", s, perl = TRUE)
      if (is_hdr) { in_sec <- grepl(section, lines[[i]], fixed = TRUE); next }
      keep[[i]] <- in_sec
    }
    lines <- lines[keep]
  }
  cand <- grep(match, lines, fixed = TRUE, value = TRUE)
  cand <- cand[grepl("ms", cand, fixed = TRUE)]
  if (!is.null(value_label) && nzchar(value_label))
    cand <- cand[grepl(value_label, cand, fixed = TRUE)]
  if (length(cand) == 0L)
    return(list(ms = NA_real_, throughput = NA_real_, unit = NA_character_))
  line <- cand[[length(cand)]]
  ms <- suppressWarnings(as.numeric(sub(".*?([0-9.]+)\\s*ms.*", "\\1", line)))
  if (!is.null(value_label) && nzchar(value_label)) {
    rx <- gsub("([().])", "\\\\\\1", value_label)
    tp <- suppressWarnings(as.numeric(sub(sprintf(".*?([0-9.]+)\\s*%s.*", rx), "\\1", line)))
    unit <- if (grepl("TOPS", value_label, ignore.case = TRUE)) "TOPS" else "GFLOPS"
  } else {
    m <- regmatches(line, regexec("([0-9][0-9,.]*)\\s*(GFLOPS|TOPS)",
                                  line, perl = TRUE, ignore.case = TRUE))[[1]]
    if (length(m) >= 3L) { tp <- as.numeric(gsub(",", "", m[[2]], fixed = TRUE)); unit <- toupper(m[[3]]) }
    else { tp <- NA_real_; unit <- NA_character_ }
  }
  list(ms = ms, throughput = tp, unit = unit)
}

# ---- Oracle 3: rebaseline/clock_lock_sweep (verbatim) -----------------
oracle_rebaseline <- function(out, match, label) {
  cand <- grep(match, out, fixed = TRUE, value = TRUE)
  cand <- cand[grepl(label, cand, fixed = TRUE) & grepl("ms", cand, fixed = TRUE)]
  if (length(cand) == 0L) return(list(ms = NA_real_, tput = NA_real_))
  line <- cand[[length(cand)]]
  label_rx <- gsub("([().])", "\\\\\\1", label)
  list(
    ms   = suppressWarnings(as.numeric(sub(".*?([0-9.]+)\\s*ms.*", "\\1", line))),
    tput = suppressWarnings(as.numeric(sub(sprintf(".*?([0-9.]+)\\s*%s.*", label_rx), "\\1", line)))
  )
}

fx <- function(name) readLines(test_path("fixtures", paste0(name, ".txt")), warn = FALSE)

# bench_regress configs (pick = first) — the real per-config directives
# from data/baselines.json.
br_cases <- list(
  list(fx = "hgemm_2048",         match = "hgemm_16warp (128x128 2blk/SM)", section = NULL,  value_label = NULL),
  list(fx = "igemm_sparse_2048",  match = "igemm_sparse_tiled",             section = NULL,  value_label = "dense-equiv GFLOPS"),
  list(fx = "igemm_cpasync_4096", match = "igemm_cpasync",                  section = NULL,  value_label = NULL),
  list(fx = "flash_br16_1024",    match = "flash_attn_br16_regpv",          section = NULL,  value_label = NULL),
  list(fx = "conv2d_implicit",    match = "Implicit (single kern)",         section = "SD 64", value_label = NULL)
)

test_that("parse_throughput reproduces bench_regress (.pick_line/.parse_line, pick=first)", {
  for (c in br_cases) {
    lines <- fx(c$fx)
    want <- oracle_bench_regress(lines, match = c$match, section = c$section, value_label = c$value_label)
    got  <- parse_throughput(lines, match = c$match, section = c$section,
                             value_label = c$value_label, pick = "first")
    expect_equal(got$ms,         want$ms,         info = c$fx)
    expect_equal(got$throughput, want$throughput, info = c$fx)
    expect_equal(got$unit,       want$unit,       info = c$fx)
    expect_false(is.na(got$throughput), info = paste(c$fx, "throughput parsed"))
  }
})

test_that("parse_throughput reproduces grid_measure (parse_bench_line, pick=last)", {
  for (c in br_cases) {
    lines <- fx(c$fx)
    want <- oracle_grid(lines, match = c$match, value_label = c$value_label, section = c$section)
    got  <- parse_throughput(lines, match = c$match, section = c$section,
                             value_label = c$value_label, pick = "last")
    expect_equal(got$ms,         want$ms,         info = c$fx)
    expect_equal(got$throughput, want$throughput, info = c$fx)
    expect_equal(got$unit,       want$unit,       info = c$fx)
  }
})

test_that("parse_throughput reproduces rebaseline/clock_lock (parse_bench_line, pick=last)", {
  rb_cases <- list(
    list(fx = "hgemm_2048",        match = "hgemm_16warp (128x128 2blk/SM)", label = "GFLOPS"),
    list(fx = "igemm_sparse_2048", match = "igemm_sparse_tiled",             label = "dense-equiv GFLOPS")
  )
  for (c in rb_cases) {
    lines <- fx(c$fx)
    want <- oracle_rebaseline(lines, match = c$match, label = c$label)
    got  <- parse_throughput(lines, match = c$match, value_label = c$label, pick = "last")
    expect_equal(got$ms,         want$ms,   info = c$fx)
    expect_equal(got$throughput, want$tput, info = c$fx)
  }
})

test_that("bench_implicit_gemm --per-launch lines are invisible to every conv2d selector (#223)", {
  # Fixture: raw stdout of `bench_implicit_gemm --per-launch` at 5dc8901,
  # 2026-11-05. FORMAT FIXTURE ONLY: one uncontrolled smoke run, with no
  # pre-registration and no peer hold, captured while another session's R
  # jobs held the host at ~88% CPU. Its timings are not evidence for #223
  # and must not be cited.
  # The mode adds per-launch* lines (per-launch, -summary, -phase, -config)
  # inside each config section. They must never be selected: the gate
  # (pick = first) and conv2d_variance.R must read the aggregate LINE in
  # both modes. It is the same line, not the same measurement: in
  # --per-launch mode its window also holds the extra events, so a
  # consumer that stores it must record the mode.
  lines <- fx("conv2d_implicit_per_launch")
  is_per_launch <- grepl("^\\s*per-launch", lines)
  expect_gt(sum(is_per_launch), 0)
  # All four kinds are present, so a fixture recaptured from a bench that
  # dropped one cannot pass silently.
  for (kind in c("per-launch config=", "per-launch-summary config=",
                 "per-launch-phase config=", "per-launch-config config=")) {
    expect_true(any(startsWith(trimws(lines), kind)), info = kind)
  }
  # Neither parser match string appears in a per-launch line.
  expect_false(any(grepl("Explicit (im2col+GEMM)", lines[is_per_launch], fixed = TRUE)))
  expect_false(any(grepl("Implicit (single kern)", lines[is_per_launch], fixed = TRUE)))
  # By construction the lines carry neither filter token.
  expect_false(any(grepl("ms", lines[is_per_launch], fixed = TRUE)))
  expect_false(any(grepl("GFLOPS|TOPS", lines[is_per_launch], ignore.case = TRUE)))
  expect_false(any(grepl("^(===|---)", trimws(lines[is_per_launch]))))

  stripped <- lines[!is_per_launch]
  selectors <- list(
    c("SD 64",  "Explicit (im2col+GEMM)"), c("SD 64",  "Implicit (single kern)"),
    c("SD 32",  "Explicit (im2col+GEMM)"), c("SD 32",  "Implicit (single kern)"),
    c("SD 128", "Explicit (im2col+GEMM)"), c("SD 128", "Implicit (single kern)")
  )
  for (s in selectors) for (pick in c("first", "last")) {
    label <- paste(s[1], s[2], pick)
    got  <- parse_throughput(lines,    match = s[2], section = s[1], pick = pick)
    want <- parse_throughput(stripped, match = s[2], section = s[1], pick = pick)
    expect_identical(got, want, info = label)
    expect_false(is.na(got$throughput), info = label)
    expect_true(startsWith(trimws(got$line), s[2]), info = label)
  }
  # The legacy bench_regress oracle agrees on the gate's own selector.
  gate <- oracle_bench_regress(lines, match = "Implicit (single kern)", section = "SD 64")
  expect_equal(parse_throughput(lines, match = "Implicit (single kern)",
                                section = "SD 64", pick = "first")$throughput,
               gate$throughput)
})

test_that("igemm_sparse value_label selects dense-equiv (second number), not eff", {
  lines <- fx("igemm_sparse_2048")
  got <- parse_throughput(lines, match = "igemm_sparse_tiled",
                          value_label = "dense-equiv GFLOPS", pick = "first")
  # Fixture line: '... 19531 eff GFLOPS  39062 dense-equiv GFLOPS'
  expect_equal(got$throughput, 39062)
  expect_equal(got$unit, "GFLOPS")
})

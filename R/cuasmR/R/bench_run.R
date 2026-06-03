# bench_run.R -- run a benchmark executable once, snapshotting GPU state.
#
# The single per-sample primitive shared by every measurement harness
# (bench_regress.R, grid_measure.R, rebaseline_measure.R,
# clock_lock_sweep.R). Migrated from those scripts' near-identical
# run_bench() / run_one_sample() / inline-runner copies (issue #134).

#' Run a benchmark executable once, snapshotting GPU state around it.
#'
#' Captures \code{\link{capture_gpu_state}()} immediately before and
#' after the launch, so a caller can later refuse to compare a run taken
#' under throttle / on battery / at the wrong clock.
#'
#' The benchmark resolves its cubin with a \emph{cwd-relative} filename
#' (\code{cuModuleLoad}), so it must run from the executable's own
#' directory. This function does \strong{not} change the working
#' directory -- the caller is responsible for \code{setwd()} into the exe
#' dir (typically once around a warmup + sample loop). Pass an absolute
#' \code{exe} path so the launch itself is cwd-independent.
#'
#' @param exe Path to the benchmark executable (absolute recommended).
#' @param args Character vector of command-line arguments (coerced with
#'   \code{as.character}).
#' @param timeout Seconds before the child is killed; \code{0} (default)
#'   means no limit, matching base \code{\link{system2}}.
#' @return \code{list(out, pre, post, rc)} where \code{out} is the
#'   captured stdout+stderr \emph{line vector}, \code{pre}/\code{post}
#'   are \code{capture_gpu_state()} snapshots (\code{NULL} off-GPU), and
#'   \code{rc} is the integer exit status (\code{0} on success). A
#'   benchmark that catches \code{SIGINT} exits \code{130}; callers that
#'   want single-Ctrl+C cancellation should test \code{rc == 130L}.
#'   \code{out} stays a line vector (not a collapsed string) so the
#'   throughput parser can scan candidate lines.
#' @export
run_bench <- function(exe, args, timeout = 0) {
  pre <- capture_gpu_state()
  out <- tryCatch(
    suppressWarnings(system2(exe, as.character(args),
                             stdout = TRUE, stderr = TRUE, timeout = timeout)),
    error = function(e) {
      structure(character(0), status = 1L)
    }
  )
  post <- capture_gpu_state()
  status <- attr(out, "status")
  list(out = out, pre = pre, post = post,
       rc = if (is.null(status)) 0L else as.integer(status))
}

#' Parse ms + throughput from a benchmark's stdout.
#'
#' A bench prints one \code{<X> ms ... <Y> (GFLOPS|TOPS)} line per kernel
#' variant; this picks the right one and extracts the timing + throughput.
#' Consolidates the three near-identical \code{parse_bench_line()} copies
#' that lived in grid_measure.R / rebaseline_measure.R / clock_lock_sweep.R
#' and the \code{.pick_line}/\code{.parse_line} pair in bench_regress.R
#' (issue #134).
#'
#' Line selection, in order: keep only lines after a \code{section}
#' header (a line whose trimmed form starts \code{===} or \code{---}) up
#' to the next header, if \code{section} is given; keep lines containing
#' \code{match} (fixed substring) if given; keep lines containing
#' \code{"ms"}; then keep lines carrying \code{value_label} (if given)
#' else a \code{GFLOPS}/\code{TOPS} token. From the survivors take the
#' \code{first} or \code{last} per \code{pick}.
#'
#' @param lines Character \emph{vector} of bench stdout lines (the
#'   \code{out} element of \code{\link{run_bench}()}). Not a collapsed
#'   string -- the scan is line-wise.
#' @param match Optional fixed substring identifying the kernel's line.
#' @param section Optional fixed substring of a section header bracketing
#'   the search.
#' @param value_label Optional text immediately after the throughput
#'   number (e.g. \code{"dense-equiv GFLOPS"}); required when the line
#'   carries multiple numbers so the right column is taken.
#' @param pick \code{"first"} (bench_regress legacy) or \code{"last"}
#'   (the probe harnesses) candidate line.
#' @return \code{list(ms, throughput, unit, line)}; the numeric fields
#'   and \code{line} (the selected stdout line) are \code{NA} when
#'   nothing matched. \code{unit} is derived from \code{value_label}'s
#'   trailing \code{GFLOPS}/\code{TOPS} word, or the matched token,
#'   else \code{"GFLOPS"}.
#' @export
parse_throughput <- function(lines, match = NULL, section = NULL,
                             value_label = NULL, pick = c("first", "last")) {
  pick <- match.arg(pick)
  na_out <- list(ms = NA_real_, throughput = NA_real_,
                 unit = NA_character_, line = NA_character_)

  # Section filter -- same header heuristic as the original parsers.
  if (!is.null(section) && nzchar(section)) {
    in_sec <- FALSE
    keep <- logical(length(lines))
    for (i in seq_along(lines)) {
      if (grepl("^(===|---)", trimws(lines[[i]]), perl = TRUE)) {
        in_sec <- grepl(section, lines[[i]], fixed = TRUE)
        next
      }
      keep[[i]] <- in_sec
    }
    lines <- lines[keep]
  }

  cand <- if (is.null(match) || !nzchar(match)) lines
          else grep(match, lines, fixed = TRUE, value = TRUE)
  cand <- cand[grepl("ms", cand, fixed = TRUE)]
  if (!is.null(value_label) && nzchar(value_label)) {
    cand <- cand[grepl(value_label, cand, fixed = TRUE)]
  } else {
    cand <- cand[grepl("GFLOPS|TOPS", cand, ignore.case = TRUE)]
  }
  if (!length(cand)) return(na_out)

  line <- if (pick == "first") cand[[1]] else cand[[length(cand)]]

  ms <- suppressWarnings(as.numeric(
    sub(".*?([0-9.]+)\\s*ms.*", "\\1", line)))

  if (!is.null(value_label) && nzchar(value_label)) {
    rx <- gsub("([][}{()|.+*?^$\\\\])", "\\\\\\1", value_label, perl = TRUE)
    m <- regmatches(line, regexec(sprintf("([0-9][0-9,.]*)\\s*%s", rx),
                                  line, perl = TRUE, ignore.case = TRUE))[[1]]
    tp <- if (length(m) >= 2L)
            as.numeric(gsub(",", "", m[[2]], fixed = TRUE)) else NA_real_
    u <- regmatches(value_label, regexec("(GFLOPS|TOPS)\\b", value_label,
                                         perl = TRUE, ignore.case = TRUE))[[1]]
    unit <- if (length(u) >= 2L) toupper(u[[2]]) else "GFLOPS"
  } else {
    m <- regmatches(line, regexec("([0-9][0-9,.]*)\\s*(GFLOPS|TOPS)",
                                  line, perl = TRUE, ignore.case = TRUE))[[1]]
    if (length(m) >= 3L) {
      tp <- as.numeric(gsub(",", "", m[[2]], fixed = TRUE))
      unit <- toupper(m[[3]])
    } else {
      tp <- NA_real_
      unit <- NA_character_
    }
  }
  list(ms = ms, throughput = tp, unit = unit, line = line)
}

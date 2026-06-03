# bench_io.R -- append-only JSONL store for benchmark samples (issue #134).
#
# One JSON object per line: a Ctrl+C between rows leaves earlier rows
# intact, and a kill mid-row leaves at most one truncated final line that
# the tolerant reader drops. Shared by grid_measure.R (writer + resume-key
# reader) and grid_collect.R (materialiser).

#' Append one row to a JSONL file as a single line.
#'
#' \code{cat(..., append = TRUE)} is atomic at line boundaries on POSIX
#' (and on Windows for short writes below the pipe/page size), which is
#' what makes the store crash-safe: a hard kill can only corrupt the
#' final line, never an earlier row.
#'
#' @param jsonl_path Path to the JSONL file (created/extended).
#' @param row A named list serialised with \code{jsonlite::toJSON}
#'   (\code{auto_unbox = TRUE}, \code{na/null = "null"}, full precision).
#' @return Invisibly \code{NULL}.
#' @export
append_jsonl_row <- function(jsonl_path, row) {
  json <- jsonlite::toJSON(row, auto_unbox = TRUE, na = "null",
                           null = "null", digits = NA)
  cat(json, "\n", sep = "", file = jsonl_path, append = TRUE)
  invisible(NULL)
}

#' Read a JSONL file tolerantly, dropping unparseable lines.
#'
#' Parses each line independently; a partially-written final line (the
#' documented hard-kill failure mode) is silently dropped and counted, so
#' the rest of the store stays usable. Callers map \code{rows} to whatever
#' they need (resume keys, a data.table) and report \code{n_bad} in their
#' own voice.
#'
#' @param path Path to the JSONL file. A missing or empty file yields an
#'   empty result (no error) — callers that require the file should check
#'   first.
#' @param simplify Passed to \code{jsonlite::fromJSON} as
#'   \code{simplifyVector}: \code{FALSE} keeps nested lists (for
#'   field-by-field access), \code{TRUE} flattens (for row binding).
#' @return \code{list(rows, n_total, n_bad)}: \code{rows} is the list of
#'   successfully parsed objects, \code{n_total} the line count,
#'   \code{n_bad} the number that failed to parse.
#' @export
read_jsonl <- function(path, simplify = TRUE) {
  empty <- list(rows = list(), n_total = 0L, n_bad = 0L)
  if (is.null(path) || !file.exists(path)) return(empty)
  lines <- readLines(path, warn = FALSE)
  if (length(lines) == 0L) return(empty)
  parsed <- lapply(lines, function(l) {
    tryCatch(jsonlite::fromJSON(l, simplifyVector = simplify),
             error = function(e) NULL)
  })
  ok <- !vapply(parsed, is.null, logical(1))
  list(rows = parsed[ok], n_total = length(lines), n_bad = sum(!ok))
}

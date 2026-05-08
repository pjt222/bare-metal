#!/usr/bin/env Rscript
# sass_histogram.R - per-kernel SASS instruction histogram.
#
# Walks all *.sm_86.cubin in the repo (excluding test_/debug paths),
# disassembles via `cuobjdump -sass`, counts opcodes by family, emits
# both CSV (machine readable) and a Markdown table (human readable).
#
# Usage:
#   Rscript scripts/sass_histogram.R                    # writes docs/sass_histogram.{csv,md}
#   Rscript scripts/sass_histogram.R --kernel HGEMM     # filter by name substring
#   Rscript scripts/sass_histogram.R --root phase3      # restrict scan to subtree
#   Rscript scripts/sass_histogram.R --quiet            # suppress per-cubin progress
#
# Categories: each instruction counted exactly once into the first
# matching category, then 'other' catches the rest.
#   Tensor cores:    HMMA, IMMA
#   Smem:            LDSM (TC vector loads), LDS, STS
#   Async:           LDGSTS (cp.async)
#   Global:          LDG, STG
#   FP scalar:       FFMA, FADD, FMUL, FSEL
#   Special:         MUFU (sin/cos/exp/rsq), SHFL
#   Integer:         IADD3, IMAD, ISETP, LOP3
#   Control:         BRA, EXIT, BAR, BARRIER, WARPSYNC, NOP, S2R
#
# useful_pct = (HMMA + IMMA + FFMA + FMUL + FADD) / total
# Kernels with low useful_pct are bookkeeping-bound (target for SASS hand-tune).

# (no library() loads needed — base R only)

# ---------- argument parsing (base R, no deps) ----------
parse_args <- function(argv) {
  defaults <- list(
    root    = ".",
    kernel  = NULL,
    csv_out = "docs/sass_histogram.csv",
    md_out  = "docs/sass_histogram.md",
    quiet   = FALSE
  )
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if (a == "--root")        { defaults$root    <- argv[i+1]; i <- i + 2 }
    else if (a == "--kernel") { defaults$kernel  <- argv[i+1]; i <- i + 2 }
    else if (a == "--csv-out"){ defaults$csv_out <- argv[i+1]; i <- i + 2 }
    else if (a == "--md-out") { defaults$md_out  <- argv[i+1]; i <- i + 2 }
    else if (a == "--quiet")  { defaults$quiet   <- TRUE;      i <- i + 1 }
    else if (a == "--help" || a == "-h") {
      cat("Usage: sass_histogram.R [--root DIR] [--kernel SUBSTR]",
          "                        [--csv-out PATH] [--md-out PATH] [--quiet]",
          sep = "\n")
      quit(status = 0)
    }
    else { stop("unknown arg: ", a) }
  }
  defaults
}

# ---------- categorisation ----------
# Order matters: first match wins. Patterns applied to opcode string
# extracted from disassembly text (see LINE_RE).
CATEGORIES <- list(
  c("HMMA",     "^HMMA\\b"),
  c("IMMA",     "^IMMA\\b"),
  c("LDSM",     "^LDSM\\b"),
  c("LDS",      "^LDS(?!M)\\b"),       # LDS but not LDSM
  c("STS",      "^STS\\b"),
  c("LDGSTS",   "^LDGSTS\\b"),         # cp.async
  c("LDG",      "^LDG(?!STS)\\b"),
  c("STG",      "^STG\\b"),
  c("FFMA",     "^FFMA\\b"),
  c("FADD",     "^FADD\\b"),
  c("FMUL",     "^FMUL\\b"),
  c("FSEL",     "^FSEL\\b"),
  c("MUFU",     "^MUFU"),
  c("SHFL",     "^SHFL"),
  c("IADD3",    "^IADD3\\b"),
  c("IMAD",     "^IMAD\\b"),
  c("ISETP",    "^ISETP\\b"),
  c("LOP3",     "^LOP3\\b"),
  c("BRA",      "^BRA\\b"),
  c("EXIT",     "^EXIT\\b"),
  c("BAR",      "^BAR\\b"),
  c("BARRIER",  "^BARRIER\\b"),
  c("WARPSYNC", "^WARPSYNC\\b"),
  c("NOP",      "^NOP\\b"),
  c("S2R",      "^S2R\\b")
)
CAT_NAMES <- vapply(CATEGORIES, `[`, character(1), 1)
USEFUL <- c("HMMA", "IMMA", "FFMA", "FMUL", "FADD")

# Strip leading address/encoding from cuobjdump output.
# Sample line:
#   "        /*0250*/                   LDG.E R28, [R2.64] ;"
# We extract the opcode token (LDG.E here).
# Optional predicate prefix (@P0, @!P3) is also tolerated.
LINE_RE <- "^\\s*/\\*[0-9a-f]+\\*/\\s+(?:@!?P[0-9]+\\s+)?(\\S+)"

categorize_kernel <- function(sass_lines) {
  counts <- setNames(integer(length(CAT_NAMES) + 1), c(CAT_NAMES, "other"))
  total <- 0L

  m <- regmatches(sass_lines, regexec(LINE_RE, sass_lines, perl = TRUE))
  for (mm in m) {
    if (length(mm) < 2) next
    opcode <- mm[2]
    total <- total + 1L
    bucketed <- FALSE
    for (cat in CATEGORIES) {
      if (grepl(cat[2], opcode, perl = TRUE)) {
        counts[cat[1]] <- counts[cat[1]] + 1L
        bucketed <- TRUE
        break
      }
    }
    if (!bucketed) counts["other"] <- counts["other"] + 1L
  }
  list(counts = counts, total = total)
}

# ---------- cuobjdump driver ----------
disasm_cubin <- function(cubin_path) {
  out <- tryCatch(
    system2("cuobjdump", c("-sass", cubin_path), stdout = TRUE, stderr = FALSE),
    error = function(e) {
      message("  warn: cuobjdump failed on ", cubin_path, ": ", conditionMessage(e))
      character(0)
    }
  )
  if (length(out) == 0) return(list())

  # Split into per-function blocks.
  func_re <- "^\\s*Function\\s*:\\s*(\\S+)"
  hits <- regmatches(out, regexec(func_re, out, perl = TRUE))
  starts <- which(vapply(hits, length, integer(1)) >= 2L)
  if (!length(starts)) return(list())

  names_v <- vapply(hits[starts], `[`, character(1), 2)
  ends <- c(starts[-1] - 1L, length(out))
  kernels <- list()
  for (i in seq_along(starts)) {
    kernels[[names_v[i]]] <- out[(starts[i] + 1L):ends[i]]
  }
  kernels
}

find_cubins <- function(root) {
  paths <- list.files(root, pattern = "\\.sm_86\\.cubin$",
                      recursive = TRUE, full.names = TRUE)
  # Drop test_/debug paths.
  rels <- substring(paths, nchar(root) + 2L)   # strip leading root/
  exclude <- grepl("(^|/)(test_|debug)", rels)
  paths[!exclude]
}

# ---------- main ----------
main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  args <- parse_args(argv)

  root <- normalizePath(args$root, mustWork = TRUE)
  cubins <- sort(find_cubins(root))
  rows <- list()

  for (cubin in cubins) {
    if (!args$quiet) {
      cat(sprintf("  scanning %s ...\n", substring(cubin, nchar(root) + 2L)))
    }
    kernels <- disasm_cubin(cubin)
    for (kname in names(kernels)) {
      if (!is.null(args$kernel) &&
          !grepl(args$kernel, kname, ignore.case = TRUE, fixed = FALSE)) next
      r <- categorize_kernel(kernels[[kname]])
      if (r$total == 0L) next
      useful <- sum(r$counts[USEFUL])
      useful_pct <- round(100 * useful / r$total, 1)
      row <- c(
        list(cubin = substring(cubin, nchar(root) + 2L),
             kernel = kname,
             total_inst = r$total),
        as.list(r$counts),
        list(useful_pct = useful_pct)
      )
      rows[[length(rows) + 1L]] <- row
    }
  }

  if (!length(rows)) {
    cat("No kernels matched.\n")
    return(invisible())
  }

  # Build data frame with stable column order.
  col_order <- c("cubin", "kernel", "total_inst", CAT_NAMES, "other", "useful_pct")
  df <- do.call(rbind, lapply(rows, function(r) {
    as.data.frame(r[col_order], stringsAsFactors = FALSE)
  }))
  df <- df[order(-df$useful_pct, df$cubin), , drop = FALSE]

  csv_path <- file.path(root, args$csv_out)
  dir.create(dirname(csv_path), showWarnings = FALSE, recursive = TRUE)
  # Match Python csv.DictWriter format: no quotes around string fields
  # unless they contain a comma. Base R `write.csv` always quotes strings,
  # so we use `write.table` with `quote = FALSE` instead.
  write.table(df, csv_path, sep = ",", row.names = FALSE,
              quote = FALSE, qmethod = "double")
  cat(sprintf("\nWrote %d rows to %s\n", nrow(df), csv_path))

  # Markdown summary
  md_cols <- c("kernel", "total_inst", "HMMA", "IMMA",
               "LDSM", "LDGSTS", "FFMA", "MUFU", "IMAD", "BRA", "useful_pct")
  md_path <- file.path(root, args$md_out)
  con <- file(md_path, "w")
  writeLines(c(
    "# SASS Instruction Histogram",
    "",
    "Auto-generated by `scripts/sass_histogram.R`.",
    "",
    "`useful_pct` = (HMMA + IMMA + FFMA + FMUL + FADD) / total_inst.",
    "Sort: highest `useful_pct` first.",
    "",
    paste0("| ", paste(md_cols, collapse = " | "), " |"),
    paste0("|", paste(ifelse(md_cols == "kernel", "---", "---:"), collapse = "|"), "|")
  ), con)
  for (i in seq_len(nrow(df))) {
    cells <- vapply(md_cols, function(c) {
      v <- df[[c]][i]
      if (c == "useful_pct") sprintf("%.1f%%", v) else as.character(v)
    }, character(1))
    writeLines(paste0("| ", paste(cells, collapse = " | "), " |"), con)
  }
  close(con)
  cat(sprintf("Wrote markdown summary to %s\n", md_path))
}

if (sys.nframe() == 0L) main()

#!/usr/bin/env Rscript
# reg_audit.R - Per-kernel register pressure and spill audit (#91).
#
# Walks every .sm_86.cubin in the repo, runs `cuobjdump --dump-resource-usage`,
# parses register count, shared-memory bytes, and local-memory bytes
# (spill indicator). Cross-references the source .cu file for
# __launch_bounds__ to compute per-SM occupancy. Emits CSV + Markdown.
#
# Usage:
#   Rscript scripts/reg_audit.R                    # default outputs
#   Rscript scripts/reg_audit.R --root phase3
#   Rscript scripts/reg_audit.R --csv-out PATH --md-out PATH
#
# Notes on SHARED column:
#   cuobjdump reports STATIC __shared__ only. Kernels using
#   `extern __shared__` (dynamic smem) appear as SHARED:0 even when
#   they use 45+ KB at launch. The audit flags such kernels via the
#   smem_dynamic column when the source contains `extern __shared__`.

# (uses base R only -- no library() loads needed)

# ----------------------------------------------------------------------
# GA104 sm_86 hardware constants
# ----------------------------------------------------------------------
HW_REGS_PER_SM      <- 65536L      # 64 K registers / SM
HW_THREADS_PER_SM   <- 1536L
HW_WARPS_PER_SM     <- 48L         # 1536 / 32
HW_MAX_BLOCKS_SM    <- 16L
HW_SMEM_PER_SM_KB   <- 100L        # max dynamic smem on sm_86
HW_STATIC_SMEM_KB   <- 48L         # default static smem cap

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list(
    root    = ".",
    csv_out = "docs/register_audit.csv",
    md_out  = "docs/register_audit.md",
    quiet   = FALSE
  )
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if      (a == "--root")    { out$root    <- argv[i+1]; i <- i + 2 }
    else if (a == "--csv-out") { out$csv_out <- argv[i+1]; i <- i + 2 }
    else if (a == "--md-out")  { out$md_out  <- argv[i+1]; i <- i + 2 }
    else if (a == "--quiet")   { out$quiet   <- TRUE;      i <- i + 1 }
    else if (a %in% c("-h", "--help")) {
      cat("Usage: reg_audit.R [--root DIR] [--csv-out PATH] [--md-out PATH] [--quiet]\n")
      quit(status = 0)
    }
    else stop("unknown arg: ", a)
  }
  out
}

# ----------------------------------------------------------------------
# Find cubins (skip test_/debug subtrees)
# ----------------------------------------------------------------------
find_cubins <- function(root) {
  paths <- list.files(root, pattern = "\\.sm_86\\.cubin$",
                      recursive = TRUE, full.names = TRUE)
  rels <- substring(paths, nchar(normalizePath(root)) + 2L)
  excl <- grepl("(^|/)(test_|debug)", rels)
  paths[!excl]
}

# ----------------------------------------------------------------------
# Parse cuobjdump --dump-resource-usage. Returns data.frame with one
# row per `Function NAME:` block found in the cubin.
# ----------------------------------------------------------------------
parse_resource_usage <- function(cubin_path) {
  out <- tryCatch(
    suppressWarnings(system2("cuobjdump",
                             c("--dump-resource-usage", shQuote(cubin_path)),
                             stdout = TRUE, stderr = FALSE)),
    error = function(e) character(0)
  )
  if (!length(out)) return(NULL)

  rows <- list()
  current_name <- NULL
  fn_re <- "^\\s*Function\\s+(\\S+):\\s*$"
  for (ln in out) {
    m <- regmatches(ln, regexec(fn_re, ln, perl = TRUE))[[1]]
    if (length(m) >= 2) {
      current_name <- m[2]
      next
    }
    if (is.null(current_name)) next
    # Match the resource line:
    # "  REG:64 STACK:0 SHARED:37888 LOCAL:0 CONSTANT[0]:388 TEXTURE:0 SURFACE:0 SAMPLER:0"
    if (grepl("REG:", ln, fixed = TRUE)) {
      grab <- function(key) {
        mm <- regmatches(ln, regexec(paste0("\\b", key, ":(\\d+)"), ln, perl = TRUE))[[1]]
        if (length(mm) >= 2) as.integer(mm[2]) else NA_integer_
      }
      rows[[length(rows) + 1L]] <- data.frame(
        kernel       = current_name,
        regs         = grab("REG"),
        stack_bytes  = grab("STACK"),
        smem_static  = grab("SHARED"),
        local_bytes  = grab("LOCAL"),
        const_bytes  = grab("CONSTANT\\[0\\]"),
        stringsAsFactors = FALSE
      )
      current_name <- NULL
    }
  }
  if (!length(rows)) return(NULL)
  do.call(rbind, rows)
}

# ----------------------------------------------------------------------
# Best-effort: extract __launch_bounds__(BLOCK_SIZE [, MIN_BLOCKS]) for
# the named kernel from a sibling .cu source file. Returns a list with
# block_size and min_blocks (NA if not found / not parseable).
# ----------------------------------------------------------------------
find_launch_bounds <- function(cubin_path, kernel_name) {
  # Try sibling .cu (drop .sm_86.cubin).
  cu_candidate <- sub("\\.sm_86\\.cubin$", ".cu", cubin_path)
  out <- list(block_size = NA_integer_, min_blocks = NA_integer_,
              source     = NA_character_)

  read_lines_safe <- function(p) {
    if (!file.exists(p)) return(NULL)
    tryCatch(readLines(p, warn = FALSE), error = function(e) NULL)
  }

  # Build a dictionary of #define NAME NUMERIC_OR_SIMPLE_EXPR from the
  # source so we can substitute symbols in __launch_bounds__ args.
  # Resolution is iterative (defines can reference other defines).
  build_defines <- function(lines) {
    pat <- "^\\s*#\\s*define\\s+([A-Za-z_][A-Za-z0-9_]*)\\s+(\\(.+\\)|[^/\\s][^/]*?)\\s*(?://.*)?$"
    defs <- list()
    for (ln in lines) {
      m <- regmatches(ln, regexec(pat, ln, perl = TRUE))[[1]]
      if (length(m) >= 3) {
        nm <- m[2]
        val <- trimws(m[3])
        # Strip surrounding parens.
        val <- sub("^\\((.+)\\)$", "\\1", val)
        defs[[nm]] <- val
      }
    }
    defs
  }

  # Substitute defines until the expression is purely numeric and
  # arithmetic (digits, ops, parens, whitespace).
  resolve_expr <- function(expr, defs, max_passes = 8L) {
    safe <- "^[0-9+\\-*/() \\t]+$"
    for (pass in seq_len(max_passes)) {
      if (grepl(safe, expr, perl = TRUE)) break
      changed <- FALSE
      for (nm in names(defs)) {
        before <- expr
        expr <- gsub(paste0("\\b", nm, "\\b"), defs[[nm]], expr, perl = TRUE)
        if (!identical(before, expr)) changed <- TRUE
      }
      if (!changed) break
    }
    expr
  }

  eval_int <- function(expr, defs) {
    if (is.null(expr) || !nzchar(expr)) return(NA_integer_)
    direct <- suppressWarnings(as.integer(expr))
    if (!is.na(direct)) return(direct)
    resolved <- resolve_expr(expr, defs)
    if (!grepl("^[0-9+\\-*/() \\t]+$", resolved, perl = TRUE)) return(NA_integer_)
    val <- tryCatch(
      eval(parse(text = resolved), envir = baseenv()),
      error = function(e) NA_real_
    )
    if (is.numeric(val) && is.finite(val)) as.integer(val) else NA_integer_
  }

  parse_one <- function(lines, src_label) {
    if (is.null(lines) || !any(grepl(kernel_name, lines, fixed = TRUE))) return(NULL)
    txt <- paste(lines, collapse = "\n")
    pat <- paste0("__launch_bounds__\\s*\\(([^,)]+)(?:,\\s*([^)]+))?\\)")
    matches <- regmatches(txt, gregexpr(pat, txt, perl = TRUE))[[1]]
    if (!length(matches)) return(NULL)
    # Pick the launch_bounds preceding the kernel decl.
    kp <- regexpr(kernel_name, txt, fixed = TRUE)[1]
    if (kp < 0) kp <- nchar(txt)
    best <- NULL
    for (m in matches) {
      mp <- regexpr(m, txt, fixed = TRUE)[1]
      if (mp < kp) best <- m
    }
    if (is.null(best)) best <- matches[1]
    parts <- regmatches(best, regexec(pat, best, perl = TRUE))[[1]]
    bs_text <- if (length(parts) >= 2) trimws(parts[2]) else NA_character_
    mb_text <- if (length(parts) >= 3) trimws(parts[3]) else ""

    defs <- build_defines(lines)
    bs_num <- eval_int(bs_text, defs)
    mb_num <- eval_int(mb_text, defs)

    list(block_size = bs_num, min_blocks = mb_num,
         block_text = bs_text, source = src_label)
  }

  # Direct sibling first.
  parsed <- parse_one(read_lines_safe(cu_candidate), basename(cu_candidate))
  if (is.null(parsed)) {
    # Sometimes a .cu compiles to a cubin with a different basename
    # (e.g. flash_br16_v2.cu -> flash_attn_br16_v2.sm_86.cubin via rename).
    # Search the immediate parent dir for any .cu mentioning the kernel.
    parent <- dirname(cubin_path)
    cus <- list.files(parent, pattern = "\\.cu$", full.names = TRUE)
    for (cu in cus) {
      parsed <- parse_one(read_lines_safe(cu), basename(cu))
      if (!is.null(parsed)) break
    }
  }
  if (is.null(parsed)) return(out)
  out$block_size <- parsed$block_size
  out$min_blocks <- parsed$min_blocks
  out$source     <- parsed$source
  out
}

# ----------------------------------------------------------------------
# Compute theoretical occupancy (blocks/SM) given the constraints.
# Uses static SHARED only (cuobjdump can't see dynamic smem).
# ----------------------------------------------------------------------
compute_occupancy <- function(regs, smem_static_bytes, block_size) {
  if (is.na(block_size) || block_size <= 0L) return(NA_integer_)

  by_threads <- HW_THREADS_PER_SM %/% block_size
  by_regs    <- if (regs > 0L)
                  HW_REGS_PER_SM %/% (regs * block_size)
                else HW_MAX_BLOCKS_SM
  by_smem    <- if (smem_static_bytes > 0L)
                  (HW_SMEM_PER_SM_KB * 1024L) %/% smem_static_bytes
                else HW_MAX_BLOCKS_SM

  min(c(by_threads, by_regs, by_smem, HW_MAX_BLOCKS_SM))
}

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- normalizePath(args$root, mustWork = TRUE)

  cubins <- sort(find_cubins(root))
  rows <- list()

  for (cubin in cubins) {
    if (!args$quiet) {
      cat(sprintf("  scanning %s ...\n",
                  substring(cubin, nchar(root) + 2L)))
    }
    res <- parse_resource_usage(cubin)
    if (is.null(res)) next

    for (i in seq_len(nrow(res))) {
      kn <- res$kernel[i]
      lb <- find_launch_bounds(cubin, kn)
      cu_text <- {
        cu_path <- sub("\\.sm_86\\.cubin$", ".cu", cubin)
        if (file.exists(cu_path)) paste(readLines(cu_path, warn = FALSE),
                                         collapse = "\n") else ""
      }
      smem_dynamic <- grepl("extern\\s+__shared__", cu_text, perl = TRUE)

      occ <- compute_occupancy(res$regs[i], res$smem_static[i], lb$block_size)
      regs_used_per_block <- if (!is.na(lb$block_size))
                                res$regs[i] * lb$block_size else NA_integer_

      rows[[length(rows) + 1L]] <- data.frame(
        cubin           = substring(cubin, nchar(root) + 2L),
        kernel          = kn,
        regs            = res$regs[i],
        block_size      = lb$block_size,
        regs_per_block  = regs_used_per_block,
        smem_static_kb  = round(res$smem_static[i] / 1024, 1),
        smem_dynamic    = smem_dynamic,
        local_bytes     = res$local_bytes[i],
        has_spill       = res$local_bytes[i] > 0L,
        const_bytes     = res$const_bytes[i],
        theo_blocks_sm  = occ,
        warps_per_sm    = if (!is.na(occ) && !is.na(lb$block_size))
                            occ * (lb$block_size %/% 32L) else NA_integer_,
        stringsAsFactors = FALSE
      )
    }
  }

  if (!length(rows)) {
    cat("No kernels found.\n"); return(invisible())
  }
  df <- do.call(rbind, rows)
  # Sort: spillers first, then descending regs.
  df <- df[order(-df$has_spill, -df$regs, df$cubin), , drop = FALSE]

  # Write CSV.
  csv_path <- file.path(root, args$csv_out)
  dir.create(dirname(csv_path), showWarnings = FALSE, recursive = TRUE)
  write.table(df, csv_path, sep = ",", row.names = FALSE,
              quote = FALSE, qmethod = "double")
  cat(sprintf("\nWrote %d rows to %s\n", nrow(df), csv_path))

  # Write Markdown.
  md_path <- file.path(root, args$md_out)
  con <- file(md_path, "w"); on.exit(close(con), add = TRUE)
  writeLines(c(
    "# Register Pressure & Spill Audit",
    "",
    "Auto-generated by `scripts/reg_audit.R`.",
    "",
    sprintf("Total kernels scanned: **%d**", nrow(df)),
    sprintf("Kernels with register spill (LOCAL > 0): **%d**",
            sum(df$has_spill, na.rm = TRUE)),
    sprintf("Kernels using dynamic shared memory: **%d**",
            sum(df$smem_dynamic, na.rm = TRUE)),
    "",
    "## Hardware reference (RTX 3070 Ti, GA104, sm_86)",
    "",
    sprintf("- Registers / SM: **%d** (= 64 K)", HW_REGS_PER_SM),
    sprintf("- Threads / SM:   **%d** (48 warps)", HW_THREADS_PER_SM),
    sprintf("- Max blocks / SM: **%d**", HW_MAX_BLOCKS_SM),
    sprintf("- Max smem / SM:   **%d KB** (dynamic), %d KB static",
            HW_SMEM_PER_SM_KB, HW_STATIC_SMEM_KB),
    "",
    "## Per-kernel data",
    "",
    "Sort: spillers first, then descending registers/thread.",
    "",
    paste("| kernel | regs | block_size | smem_static_KB |",
          "smem_dynamic | local | spill? | theo_blocks_SM | warps/SM |"),
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
  ), con)
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    spill_mark <- if (isTRUE(r$has_spill)) ":warning: YES" else "no"
    bs_str <- if (is.na(r$block_size)) "?" else as.character(r$block_size)
    occ_str <- if (is.na(r$theo_blocks_sm)) "?" else as.character(r$theo_blocks_sm)
    warps_str <- if (is.na(r$warps_per_sm)) "?" else as.character(r$warps_per_sm)
    writeLines(sprintf(
      "| %s | %d | %s | %.1f | %s | %d | %s | %s | %s |",
      r$kernel, r$regs, bs_str, r$smem_static_kb,
      ifelse(r$smem_dynamic, "yes", "no"),
      r$local_bytes, spill_mark, occ_str, warps_str
    ), con)
  }
  writeLines(c(
    "",
    "## Findings",
    "",
    "Kernels with `spill?` = YES need investigation: spill loads/stores",
    "are local-memory traffic that bypasses the register file. Each",
    "spill load is ~400 cycles (DRAM-class latency unless L1/L2 hit).",
    "",
    "Kernels with `smem_dynamic = yes` allocate at launch via",
    "`extern __shared__`. cuobjdump's smem_static column will read 0",
    "for those; the actual size lives in the host launch params",
    "(see corresponding `bench_*.cu`).",
    "",
    sprintf("Re-run: `Rscript scripts/reg_audit.R`")
  ), con)
  cat(sprintf("Wrote markdown summary to %s\n", md_path))
}

if (sys.nframe() == 0L) main()

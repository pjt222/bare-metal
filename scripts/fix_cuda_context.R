#!/usr/bin/env Rscript
# fix_cuda_context.R - Migrate bench .cu files from cuCtxCreate to
# cuDevicePrimaryCtxRetain. Mirrors fix_cuda_context.py.
#
# Usage:
#   Rscript scripts/fix_cuda_context.R                         # all bench files
#   Rscript scripts/fix_cuda_context.R phase2/hgemm/bench.cu   # one file

# (no library() loads needed -- base R only)

fix_cu_context <- function(path) {
  content <- paste(readLines(path, warn = FALSE), collapse = "\n")

  # Skip files already migrated.
  if (grepl("cuDevicePrimaryCtxRetain", content, fixed = TRUE)) return(NULL)

  # Pattern 1: bare CHECK_CU(cuCtxCreate(&ctx, 0, dev));
  pat1 <- "CHECK_CU\\(cuCtxCreate\\(&(\\w+),\\s*0,\\s*(\\w+)\\)\\);"
  m <- regmatches(content, regexec(pat1, content, perl = TRUE))[[1]]

  if (length(m) < 3) {
    # Pattern 2: CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    pat2 <- "CUcontext\\s+(\\w+);\\s*CHECK_CU\\(cuCtxCreate\\(&\\1,\\s*0,\\s*(\\w+)\\)\\);"
    m <- regmatches(content, regexec(pat2, content, perl = TRUE))[[1]]
  }
  if (length(m) < 3) return(NULL)

  ctx_var <- m[2]
  dev_var <- m[3]
  old_create <- m[1]

  # Build replacement (same form for both patterns since the call site is identical).
  call_old <- sprintf("CHECK_CU(cuCtxCreate(&%s, 0, %s));", ctx_var, dev_var)
  call_new <- sprintf(
    "CHECK_CU(cuDevicePrimaryCtxRetain(&%s, %s));\n    CHECK_CU(cuCtxSetCurrent(%s));",
    ctx_var, dev_var, ctx_var)
  new_create <- sub(call_old, call_new, old_create, fixed = TRUE)

  content <- sub(old_create, new_create, content, fixed = TRUE)

  # Replace destroy (all occurrences).
  destroy_old <- sprintf("cuCtxDestroy(%s);", ctx_var)
  destroy_new <- sprintf("cuDevicePrimaryCtxRelease(%s);", dev_var)
  content <- gsub(destroy_old, destroy_new, content, fixed = TRUE)

  writeLines(content, path)
  c(ctx = ctx_var, dev = dev_var)
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) && argv[1] %in% c("-h", "--help")) {
    cat("Usage: fix_cuda_context.R [FILE ...]   (default: all phase*/**/bench*.cu)\n")
    quit(status = 0)
  }
  files <- if (length(argv)) argv else {
    fs1 <- list.files(".", pattern = "^bench(_.*)?\\.cu$",
                      recursive = TRUE, full.names = FALSE)
    keep <- grepl("^phase", fs1) &
            !grepl("bench_refactored", fs1, fixed = TRUE) &
            !grepl("\\.bak", fs1)
    sort(fs1[keep])
  }

  for (f in files) {
    res <- tryCatch(fix_cu_context(f),
                    error = function(e) {
                      cat(sprintf("ERROR processing %s: %s\n", f, conditionMessage(e)))
                      NULL
                    })
    if (!is.null(res)) {
      cat(sprintf("Fixed: %s (ctx=%s, dev=%s)\n", f, res["ctx"], res["dev"]))
    }
  }
}

if (sys.nframe() == 0L) main()

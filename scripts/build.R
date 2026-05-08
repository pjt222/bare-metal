#!/usr/bin/env Rscript
# build.R - Automate the bare-metal SASS workflow.
# Mirrors build.py.
#
#   compile   .cu     -> .cubin
#   disasm    .cubin  -> .cuasm   (human-editable)
#   assemble  .cuasm  -> .cubin   (after hand-editing)
#   roundtrip .cu     -> compile + disasm + reassemble + bytewise compare
#   all       .cu     -> compile + disasm (sets up hand-edit workflow)
#
# Usage:
#   Rscript scripts/build.R compile   phase1/vector_add.cu
#   Rscript scripts/build.R disasm    phase1/vector_add.sm_86.cubin
#   Rscript scripts/build.R assemble  phase1/vector_add_modified.cuasm
#   Rscript scripts/build.R roundtrip phase1/vector_add.cu
#   Rscript scripts/build.R all       phase1/vector_add.cu
#
# Note on language: disasm/assemble depend on the CuAssembler library
# (cloudcores/CuAssembler), which is Python-only. Those two operations
# delegate to tools/cuasm_helper.py. Everything else (compile, file
# comparison, orchestration) is pure R.

# (no library() loads needed -- base R only)

REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
  else            normalizePath(getwd())
}
CUASSEMBLER_PATH <- file.path(REPO_ROOT, "tools", "CuAssembler")
CUASM_HELPER     <- file.path(REPO_ROOT, "tools", "cuasm_helper.py")
SM_ARCH          <- "sm_86"

# WSL: CUDA tools live in /usr/local/cuda/bin
CUDA_BIN <- "/usr/local/cuda/bin"
if (dir.exists(CUDA_BIN) && !grepl(CUDA_BIN, Sys.getenv("PATH"), fixed = TRUE)) {
  Sys.setenv(PATH = paste(CUDA_BIN, Sys.getenv("PATH"), sep = ":"))
}

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
ensure_cuassembler <- function() {
  if (!dir.exists(CUASSEMBLER_PATH)) {
    cat(sprintf("ERROR: CuAssembler not found at %s\n", CUASSEMBLER_PATH))
    cat("Run: git clone https://github.com/cloudcores/CuAssembler.git tools/CuAssembler\n")
    quit(status = 1)
  }
  if (!file.exists(CUASM_HELPER)) {
    cat(sprintf("ERROR: cuasm_helper.py missing at %s\n", CUASM_HELPER))
    quit(status = 1)
  }
}

run_shell <- function(cmd, cwd = NULL) {
  cat(sprintf("  $ %s\n", cmd))
  rc <- if (is.null(cwd)) {
          system(cmd)
        } else {
          orig <- getwd()
          setwd(cwd); on.exit(setwd(orig), add = TRUE)
          system(cmd)
        }
  if (rc != 0L) {
    cat(sprintf("ERROR: Command failed with exit code %d\n", rc))
    quit(status = rc)
  }
  invisible(rc)
}

# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
cmd_compile <- function(source_path, output_path = NULL, extra_flags = "") {
  source_path <- normalizePath(source_path, mustWork = TRUE)
  if (is.null(output_path)) {
    output_path <- sub("\\.cu$", paste0(".", SM_ARCH, ".cubin"), source_path)
  }
  output_path <- normalizePath(output_path, mustWork = FALSE)

  cat(sprintf("\n[compile] %s -> %s\n", source_path, output_path))
  run_shell(sprintf('nvcc --cubin -arch=%s -O2 %s -o "%s" "%s"',
                    SM_ARCH, extra_flags, output_path, source_path),
            cwd = REPO_ROOT)
  cat(sprintf("  -> %s (%d bytes)\n", output_path, file.info(output_path)$size))
  output_path
}

cmd_disasm <- function(cubin_path, output_path = NULL) {
  ensure_cuassembler()
  cubin_path <- normalizePath(cubin_path, mustWork = TRUE)
  if (is.null(output_path)) {
    output_path <- sub("\\.cubin$", ".cuasm", cubin_path)
  }
  output_path <- normalizePath(output_path, mustWork = FALSE)

  cat(sprintf("\n[disasm] %s -> %s\n", cubin_path, output_path))
  rc <- system2("python3",
                c(shQuote(CUASM_HELPER), "disasm",
                  shQuote(cubin_path), shQuote(output_path)))
  if (rc != 0L) {
    cat(sprintf("ERROR: disasm via cuasm_helper failed (rc=%d)\n", rc))
    quit(status = rc)
  }
  cat(sprintf("  -> %s (%d bytes)\n", output_path, file.info(output_path)$size))

  # Also produce raw nvdisasm output for reference
  raw_sass_path <- sub("\\.cubin$", ".sass", cubin_path)
  sass_out <- tryCatch(
    suppressWarnings(system2("cuobjdump", c("-sass", shQuote(cubin_path)),
                             stdout = TRUE, stderr = FALSE)),
    error = function(e) NULL
  )
  if (!is.null(sass_out) && length(sass_out) &&
      (is.null(attr(sass_out, "status")) ||
       attr(sass_out, "status") == 0L)) {
    writeLines(sass_out, raw_sass_path)
    cat(sprintf("  -> %s (raw SASS reference)\n", raw_sass_path))
  }
  output_path
}

cmd_assemble <- function(cuasm_path, output_path = NULL) {
  ensure_cuassembler()
  cuasm_path <- normalizePath(cuasm_path, mustWork = TRUE)
  if (is.null(output_path)) {
    # vector_add.sm_86.cuasm -> vector_add.sm_86.reassembled.cubin
    output_path <- sub("\\.cuasm$", ".reassembled.cubin", cuasm_path)
  }
  output_path <- normalizePath(output_path, mustWork = FALSE)

  cat(sprintf("\n[assemble] %s -> %s\n", cuasm_path, output_path))
  rc <- system2("python3",
                c(shQuote(CUASM_HELPER), "assemble",
                  shQuote(cuasm_path), shQuote(output_path)))
  if (rc != 0L) {
    cat(sprintf("ERROR: assemble via cuasm_helper failed (rc=%d)\n", rc))
    quit(status = rc)
  }
  cat(sprintf("  -> %s (%d bytes)\n", output_path, file.info(output_path)$size))
  output_path
}

cmd_roundtrip <- function(source_path) {
  cat(sprintf("\n[roundtrip] Testing CuAssembler stability on %s\n", source_path))
  cat("  This compiles, disassembles, reassembles, and checks the result matches.\n\n")

  cubin_path  <- cmd_compile(source_path)
  cuasm_path  <- cmd_disasm(cubin_path)
  reassembled <- cmd_assemble(cuasm_path)

  original_size  <- file.info(cubin_path)$size
  reassembled_sz <- file.info(reassembled)$size
  cat("\n[roundtrip] Comparing cubins:\n")
  cat(sprintf("  Original:     %s (%d bytes)\n", cubin_path, original_size))
  cat(sprintf("  Reassembled:  %s (%d bytes)\n", reassembled, reassembled_sz))

  # Bytewise compare
  bytes_orig <- readBin(cubin_path,  raw(), n = original_size)
  bytes_re   <- readBin(reassembled, raw(), n = reassembled_sz)
  if (length(bytes_orig) == length(bytes_re) && all(bytes_orig == bytes_re)) {
    cat("  RESULT: IDENTICAL -- CuAssembler roundtrip is stable. Safe to hand-edit.\n")
  } else {
    cat("  RESULT: DIFFERENT -- Cubins differ.\n")
    cat("  This may be OK (CuAssembler may reorder some metadata).\n")
    cat("  Run both cubins and compare outputs to verify correctness.\n")
  }
  cuasm_path
}

cmd_all <- function(source_path) {
  cat(sprintf("\n[all] Full workflow for %s\n", source_path))
  cat("  After this completes, hand-edit the .cuasm file, then run:\n")
  cat("  Rscript scripts/build.R assemble <path_to_modified.cuasm>\n\n")
  cubin_path <- cmd_compile(source_path)
  cuasm_path <- cmd_disasm(cubin_path)
  cat(sprintf("\n  Edit: %s\n", cuasm_path))
  cat(sprintf("  Then: Rscript scripts/build.R assemble %s\n", cuasm_path))
}

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
parse_args <- function(argv) {
  if (length(argv) < 2 || argv[1] %in% c("-h", "--help")) {
    cat("Usage: build.R <command> <input> [-o OUTPUT] [--flags 'FLAGS']\n")
    cat("Commands: compile, disasm, assemble, roundtrip, all\n")
    quit(status = 0)
  }
  out <- list(command = argv[1], input = argv[2], output = NULL, flags = "")
  i <- 3
  while (i <= length(argv)) {
    a <- argv[i]
    if (a %in% c("-o", "--output")) { out$output <- argv[i + 1]; i <- i + 2 }
    else if (a == "--flags")        { out$flags  <- argv[i + 1]; i <- i + 2 }
    else stop("unknown arg: ", a)
  }
  out
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  switch(args$command,
    "compile"   = cmd_compile(args$input, args$output, args$flags),
    "disasm"    = cmd_disasm(args$input, args$output),
    "assemble"  = cmd_assemble(args$input, args$output),
    "roundtrip" = cmd_roundtrip(args$input),
    "all"       = cmd_all(args$input),
    stop("unknown command: ", args$command,
         "\nValid: compile, disasm, assemble, roundtrip, all")
  )
}

if (sys.nframe() == 0L) main()

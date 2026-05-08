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
# disasm/assemble use the cuasmR R package (R/cuasmR/, install via
#   Rscript -e 'install.packages("R/cuasmR", repos=NULL, type="source")'
# or renv::install("R/cuasmR")). cuasmR shells out to nvdisasm for SASS
# decoding and patches the cubin at the byte level for hand edits.
# Replaced upstream Python CuAssembler (#102).

library(cuasmR)

REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
  else            normalizePath(getwd())
}
SM_ARCH          <- "sm_86"

# WSL: CUDA tools live in /usr/local/cuda/bin
CUDA_BIN <- "/usr/local/cuda/bin"
if (dir.exists(CUDA_BIN) && !grepl(CUDA_BIN, Sys.getenv("PATH"), fixed = TRUE)) {
  Sys.setenv(PATH = paste(CUDA_BIN, Sys.getenv("PATH"), sep = ":"))
}

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
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

# disasm: produce a human-readable .cuasm dump alongside the original
# cubin via cuasmR. The .cuasm file lists each instruction with its
# instr_hex / ctrl_hex words; hand-edits modify those columns and the
# patched cubin is rebuilt by reading the .cuasm back, applying edits
# to the in-memory cuasm object, and calling cuasm_write().
#
# For the typical edit workflow we recommend driving cuasmR from a
# small R script (cuasm_set + cuasm_write), but the .cuasm dump is
# kept for reference and grep-friendly inspection.
cmd_disasm <- function(cubin_path, output_path = NULL) {
  cubin_path <- normalizePath(cubin_path, mustWork = TRUE)
  if (is.null(output_path)) {
    output_path <- sub("\\.cubin$", ".cuasm", cubin_path)
  }
  output_path <- normalizePath(output_path, mustWork = FALSE)

  cat(sprintf("\n[disasm] %s -> %s\n", cubin_path, output_path))
  obj <- cuasm_read(cubin_path)
  cuasm_save_cuasm(obj, output_path)
  cat(sprintf("  -> %s (%d bytes, %d kernels, %d insns)\n",
              output_path, file.info(output_path)$size,
              nrow(obj$kernels), nrow(obj$insns)))

  # Also produce raw nvdisasm/cuobjdump output for reference
  raw_sass_path <- sub("\\.cubin$", ".sass", cubin_path)
  sass_out <- suppressWarnings(
    system2("cuobjdump", c("-sass", shQuote(cubin_path)),
            stdout = TRUE, stderr = FALSE))
  if (length(sass_out) &&
      (is.null(attr(sass_out, "status")) ||
       attr(sass_out, "status") == 0L)) {
    writeLines(sass_out, raw_sass_path)
    cat(sprintf("  -> %s (raw SASS reference)\n", raw_sass_path))
  }
  output_path
}

# assemble: with cuasmR the patch path is byte-level on the original
# cubin ("compile a sibling .cu, copy the encoding, call cuasm_set").
# This sub-command is kept as a stub that simply reports the supported
# flow; we do NOT parse the .cuasm text dump back into a cubin (the
# upstream CuInsAssembler approach). The .cuasm file is reference
# material; edits drive cuasm_write() directly.
cmd_assemble <- function(cuasm_path, output_path = NULL) {
  cat("\n[assemble] cuasmR uses byte-level patching, not text-to-cubin.\n")
  cat("  To apply hand edits, write a small R script:\n\n")
  cat("      library(cuasmR)\n")
  cat("      obj <- cuasm_read(\"path/to/kernel.sm_86.cubin\")\n")
  cat("      obj <- cuasm_set(obj, kernel = \"name\", slot = N,\n")
  cat("                       instr_hex = \"0x...\", ctrl_hex = \"0x...\")\n")
  cat("      cuasm_write(obj, \"path/to/kernel.patched.cubin\")\n\n")
  cat("  See docs/cuasm_r.md for the full workflow.\n")
  invisible(NULL)
}

cmd_roundtrip <- function(source_path) {
  cat(sprintf("\n[roundtrip] cuasmR stability on %s\n", source_path))
  cat("  Compiles, reads via cuasmR, writes back, and checks byte-identical.\n\n")

  cubin_path <- cmd_compile(source_path)
  cuasm_path <- cmd_disasm(cubin_path)
  ok <- cuasm_roundtrip_check(cubin_path)
  cat(sprintf("\n[roundtrip] %s -> %s\n",
              cubin_path,
              if (ok) "BYTE-IDENTICAL (safe to hand-edit)"
              else    "DIFFERS (cuasmR bug -- file an issue)"))
  invisible(ok)
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

# Defer execution to top-level only when called as a script. The
# `cuasmR` package must already be installed (see scripts/install_cuasmR.R).
if (sys.nframe() == 0L) main()

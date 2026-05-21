#!/usr/bin/env Rscript
# verify_setup.R - Check the bare-metal GPU dev environment is ready.
# Mirrors verify_setup.py.
#
# Run from the repo root:
#   Rscript scripts/verify_setup.R

# (uses base R only — no library() loads needed)

REPO_ROOT <- {
  args_full <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args_full, value = TRUE)
  if (length(fa)) normalizePath(dirname(dirname(sub("^--file=", "", fa[1]))))
  else            normalizePath(getwd())
}


# WSL: CUDA tools live in /usr/local/cuda/bin -- prepend to PATH if missing.
CUDA_BIN <- "/usr/local/cuda/bin"
if (dir.exists(CUDA_BIN) && !grepl(CUDA_BIN, Sys.getenv("PATH"), fixed = TRUE)) {
  Sys.setenv(PATH = paste(CUDA_BIN, Sys.getenv("PATH"), sep = ":"))
}
# WSL GPU library passthrough (R hides /usr/lib/wsl/lib by default).
WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(WSL_CUDA_LIB) &&
    !grepl(WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
  cur <- Sys.getenv("LD_LIBRARY_PATH")
  Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur)) paste(WSL_CUDA_LIB, cur, sep = ":") else WSL_CUDA_LIB)
}

PASS <- "\033[92m[PASS]\033[0m"
FAIL <- "\033[91m[FAIL]\033[0m"
INFO <- "\033[94m[INFO]\033[0m"

run_command <- function(cmd, timeout = 15L) {
  out <- tryCatch(
    suppressWarnings(system(cmd, intern = TRUE, timeout = timeout, ignore.stderr = FALSE)),
    error = function(e) {
      attr(character(0), "status") <- 1L
      character(0)
    }
  )
  status <- attr(out, "status")
  list(success = is.null(status) || status == 0L,
       output  = paste(out, collapse = "\n"))
}

check_command <- function(label, cmd, expected_substring = NULL) {
  r <- run_command(cmd)
  if (!r$success) {
    cat(sprintf("%s %s\n       Command: %s\n       Output:  %s\n",
                FAIL, label, cmd, r$output))
    return(FALSE)
  }
  if (!is.null(expected_substring) &&
      !grepl(expected_substring, r$output, fixed = TRUE)) {
    cat(sprintf("%s %s -- expected '%s' in output\n       Output: %s\n",
                FAIL, label, expected_substring, r$output))
    return(FALSE)
  }
  cat(sprintf("%s %s\n", PASS, label))
  if (nzchar(r$output)) {
    first_line <- strsplit(r$output, "\n", fixed = TRUE)[[1]][1]
    cat(sprintf("       %s\n", first_line))
  }
  TRUE
}


check_cuasmR <- function() {
  ok <- requireNamespace("cuasmR", quietly = TRUE)
  if (ok) {
    ver <- as.character(packageVersion("cuasmR"))
    cat(sprintf("%s cuasmR R package installed (v%s)\n", PASS, ver))
    return(TRUE)
  }
  cat(sprintf("%s cuasmR R package not installed\n", FAIL))
  cat("       Run: Rscript scripts/install_cuasmR.R\n")
  FALSE
}

check_gpu_info <- function() {
  r <- run_command("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader")
  if (!r$success) {
    cat(sprintf("%s nvidia-smi -- GPU not detected or driver not installed\n", FAIL))
    return(FALSE)
  }
  cat(sprintf("%s GPU detected:\n", PASS))
  for (ln in strsplit(r$output, "\n", fixed = TRUE)[[1]]) {
    parts <- trimws(strsplit(ln, ",", fixed = TRUE)[[1]])
    if (length(parts) < 4) next
    name <- parts[1]; cc <- parts[2]; mem <- parts[3]; driver <- parts[4]
    cat(sprintf("       Name:             %s\n", name))
    cat(sprintf("       Compute Cap:      sm_%s\n", gsub(".", "", cc, fixed = TRUE)))
    cat(sprintf("       Memory:           %s\n", mem))
    cat(sprintf("       Driver Version:   %s\n", driver))
    if (!grepl("3070", name, fixed = TRUE) && !grepl("86", cc, fixed = TRUE)) {
      cat(sprintf("  %s Expected RTX 3070 Ti (sm_86) -- got %s (sm_%s)\n",
                  INFO, name, gsub(".", "", cc, fixed = TRUE)))
    }
  }
  TRUE
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) && argv[1] %in% c("-h", "--help")) {
    cat("Usage: verify_setup.R   (no args; runs all environment checks)\n")
    quit(status = 0)
  }
  cat(strrep("=", 60), "\n")
  cat("  bare-metal GPU -- Environment Verification\n")
  cat("  Target: RTX 3070 Ti (GA104, sm_86, Ampere)\n")
  cat(strrep("=", 60), "\n\n")

  results <- logical()

  cat("-- CUDA Toolchain --\n")
  results <- c(results,
    check_command("nvcc",      "nvcc --version",      "release"),
    check_command("cuobjdump", "cuobjdump --version", "cuobjdump"),
    check_command("nvdisasm",  "nvdisasm --version",  "nvdisasm"))
  cat("\n")

  cat("-- GPU Driver --\n")
  results <- c(results, check_gpu_info())
  cat("\n")

  cat("-- SASS Hand-Edit Toolchain --\n")
  results <- c(results, check_cuasmR())
  cat("\n")

  cat("-- sm_86 Compilation Test --\n")
  test_cu    <- file.path(REPO_ROOT, "_verify_test.cu")
  test_cubin <- file.path(REPO_ROOT, "_verify_test.cubin")
  on.exit({
    if (file.exists(test_cu))    file.remove(test_cu)
    if (file.exists(test_cubin)) file.remove(test_cubin)
  }, add = TRUE)

  writeLines('extern "C" __global__ void test_kernel(float *x) { x[threadIdx.x] = 1.0f; }',
             test_cu)
  r <- run_command(sprintf('nvcc --cubin -arch=sm_86 -o "%s" "%s" 2>&1',
                           test_cubin, test_cu))
  if (r$success && file.exists(test_cubin)) {
    cat(sprintf("%s nvcc --cubin -arch=sm_86 compiles successfully\n", PASS))
    results <- c(results, TRUE)
    r2 <- run_command(sprintf('cuobjdump -sass "%s"', test_cubin))
    sass_ok <- r2$success && (grepl("SASS", r2$output, fixed = TRUE) ||
                              grepl("code", tolower(r2$output), fixed = TRUE) ||
                              nchar(r2$output) > 10L)
    if (sass_ok) {
      cat(sprintf("%s cuobjdump disassembly works\n", PASS))
      results <- c(results, TRUE)
    } else {
      cat(sprintf("%s cuobjdump disassembly produced unexpected output\n", FAIL))
      results <- c(results, FALSE)
    }
  } else {
    cat(sprintf("%s nvcc --cubin -arch=sm_86 failed\n       %s\n", FAIL, r$output))
    results <- c(results, FALSE)
  }
  cat("\n")

  passed <- sum(results)
  total  <- length(results)
  cat(strrep("=", 60), "\n")
  if (passed == total) {
    cat(sprintf("%s All %d checks passed -- ready for bare-metal GPU work!\n\n", PASS, total))
    cat("  Next step: read kernels/tutorial/README.md and run the vector_add hello world\n")
  } else {
    cat(sprintf("%s %d/%d checks failed -- fix issues above before proceeding\n\n",
                FAIL, total - passed, total))
    cat("  See SETUP.md for installation instructions\n")
  }
  cat(strrep("=", 60), "\n")

  if (passed == total) 0L else 1L
}

if (sys.nframe() == 0L) {
  rc <- main()
  quit(status = rc)
}

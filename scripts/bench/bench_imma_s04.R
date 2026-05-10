#!/usr/bin/env Rscript
# bench_imma_s04.R -- A/B benchmark of S08->S04 IMMA hand-tunes (#96).
#
# For each target IGEMM kernel:
#   1. Bench original cubin
#   2. Swap in the *.imma_s04.sm_86.cubin variant
#   3. Re-bench
#   4. Restore original
#
# Driver: existing kernels/gemm/igemm/bench harness, parsed for the relevant
# kernel's "X ms / Y GFLOPS" line.

WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(WSL_CUDA_LIB) &&
    !grepl(WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
    cur <- Sys.getenv("LD_LIBRARY_PATH")
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                    paste(WSL_CUDA_LIB, cur, sep = ":")
                                 else WSL_CUDA_LIB)
}
Sys.setenv(PATH = paste("/usr/local/cuda/bin", Sys.getenv("PATH"), sep = ":"))

igemm_dir <- "phase2/igemm"
bench_bin <- file.path(igemm_dir, "bench")
stopifnot(file.exists(bench_bin))

# Each entry: kernel cubin file (relative to igemm_dir), regex used to grep
# its line in the bench output, and a short label.
cases <- list(
    list(cubin = "igemm_8warp_256x256.sm_86.cubin",
         label = "igemm_8warp_256x256",
         line  = "igemm_8warp_256x256"),
    list(cubin = "igemm_8warp_tribuf.sm_86.cubin",
         label = "igemm_8warp_tribuf",
         line  = "igemm_tribuf"),
    list(cubin = "igemm_8warp.sm_86.cubin",
         label = "igemm_8warp_128x128",
         line  = "igemm_8warp \\(128x128 cp.async\\)"),
    list(cubin = "igemm_pipelined.sm_86.cubin",
         label = "igemm_pipelined",
         line  = "igemm_pipelined \\(LDG"),
    list(cubin = "igemm_pipelined_cpasync.sm_86.cubin",
         label = "igemm_pipelined_cpasync",
         line  = "igemm_cpasync \\(LDGSTS dbuf\\)"),
    list(cubin = "igemm_tiled.sm_86.cubin",
         label = "igemm_tiled",
         line  = "igemm_tiled \\(64x64\\)")
)

run_bench_grep <- function(grep_re) {
    # bench loads cubins relative to its CWD; run inside phase2/igemm.
    orig_cwd <- getwd()
    setwd(igemm_dir); on.exit(setwd(orig_cwd), add = TRUE)
    out <- system2("./bench", stdout = TRUE, stderr = TRUE)
    hits <- grep(grep_re, out, value = TRUE)
    if (!length(hits)) {
        cat("WARN: no bench line matched:", grep_re, "\n")
        return(NA_real_)
    }
    line <- hits[1]
    nums <- regmatches(line,
                       regexpr("[0-9]+\\.[0-9]+\\s+(GFLOPS|TOPS)", line))
    if (!length(nums)) return(NA_real_)
    as.numeric(sub("\\s+(GFLOPS|TOPS).*$", "", nums))
}

cat(sprintf("%-25s %12s %12s %10s\n",
            "kernel", "orig GFLOPS", "patch GFLOPS", "speedup"))
cat(sprintf("%-25s %12s %12s %10s\n",
            "-----", "-----------", "------------", "-------"))

orig_dir <- igemm_dir
results <- list()
for (case in cases) {
    cubin     <- file.path(orig_dir, case$cubin)
    patched   <- sub("\\.sm_86\\.cubin$", ".imma_s04.sm_86.cubin", cubin)
    backup    <- paste0(cubin, ".bak")

    if (!file.exists(patched)) {
        cat(sprintf("%-25s missing patched cubin\n", case$label))
        next
    }

    # Run baseline: bench loads original.
    g_orig <- run_bench_grep(case$line)

    # Swap in the patched cubin.
    file.copy(cubin, backup, overwrite = TRUE)
    file.copy(patched, cubin, overwrite = TRUE)
    g_patch <- run_bench_grep(case$line)
    # Restore.
    file.copy(backup, cubin, overwrite = TRUE)
    file.remove(backup)

    speedup <- if (!is.na(g_orig) && !is.na(g_patch) && g_orig > 0)
                   g_patch / g_orig else NA
    cat(sprintf("%-25s %12.0f %12.0f %9.3fx\n",
                case$label,
                if (is.na(g_orig))  0 else g_orig,
                if (is.na(g_patch)) 0 else g_patch,
                if (is.na(speedup)) 0 else speedup))
    results[[length(results) + 1]] <- list(label = case$label,
                                           orig = g_orig, patch = g_patch,
                                           speedup = speedup)
}

speedups <- unlist(lapply(results, function(r) r$speedup))
speedups <- speedups[!is.na(speedups)]
if (length(speedups) > 0) {
    geomean <- exp(mean(log(speedups)))
    cat(sprintf("\nGeomean speedup over %d cases: %.4fx\n",
                length(speedups), geomean))
}

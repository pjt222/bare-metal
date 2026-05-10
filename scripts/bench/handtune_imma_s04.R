#!/usr/bin/env Rscript
# handtune_imma_s04.R -- Apply IMMA S08 -> S04 across IGEMM kernels via cuasmR
# (issue #96 sub-task A).
#
# For each input cubin: find every IMMA instruction whose stall field is S08,
# rewrite it to S04, write the patched cubin to *.imma_s04.cubin alongside
# the original. The bench harnesses must be modified separately to load the
# patched variant.

suppressMessages(library(cuasmR))

WSL_CUDA_LIB <- "/usr/lib/wsl/lib"
if (dir.exists(WSL_CUDA_LIB) &&
    !grepl(WSL_CUDA_LIB, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
    cur <- Sys.getenv("LD_LIBRARY_PATH")
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(cur))
                                    paste(WSL_CUDA_LIB, cur, sep = ":")
                                 else WSL_CUDA_LIB)
}

# Bit 40..43 of the 64-bit control word holds the stall count S00..S15.
ctrl_to_stall <- function(hex) {
    s   <- sub("^0x", "", tolower(hex))
    top <- strtoi(substr(s, 1, 8), 16L)
    bitwAnd(bitwShiftR(top, 8), 0xF)
}
ctrl_set_stall <- function(hex, new_stall) {
    s   <- sub("^0x", "", tolower(hex))
    top <- strtoi(substr(s, 1, 8), 16L)
    top <- bitwAnd(top, bitwNot(bitwShiftL(0xFL, 8)))
    top <- bitwOr (top, bitwShiftL(as.integer(new_stall), 8))
    sprintf("0x%08x%s", top, substr(s, 9, 16))
}

patch_imma <- function(in_path, out_path,
                       from_stall = 8L, to_stall = 4L,
                       verbose = TRUE) {
    obj  <- cuasm_read(in_path)
    rows <- which(grepl("^IMMA", obj$insns$text))
    if (!length(rows)) {
        if (verbose) cat(sprintf("  [skip] %s: no IMMA\n", basename(in_path)))
        return(invisible(NULL))
    }
    n_changed <- 0L
    for (r in rows) {
        if (ctrl_to_stall(obj$insns$ctrl_hex[r]) == from_stall) {
            obj$insns$ctrl_hex[r] <- ctrl_set_stall(obj$insns$ctrl_hex[r],
                                                    to_stall)
            n_changed <- n_changed + 1L
        }
    }
    cuasm_write(obj, out_path)
    if (verbose) cat(sprintf("  [ok] %-50s S%02d->S%02d on %d/%d IMMA -> %s\n",
                             basename(in_path), from_stall, to_stall,
                             n_changed, length(rows), basename(out_path)))
    invisible(list(n_changed = n_changed, n_imma = length(rows)))
}

# Targets: kernels with significant S08 IMMA counts identified via audit.
targets <- c(
    "kernels/gemm/igemm/igemm_8warp_256x256.sm_86.cubin",
    "kernels/gemm/igemm/igemm_8warp_tribuf.sm_86.cubin",
    "kernels/gemm/igemm/igemm_8warp.sm_86.cubin",
    "kernels/gemm/igemm/igemm_pipelined.sm_86.cubin",
    "kernels/gemm/igemm/igemm_pipelined_cpasync.sm_86.cubin",
    "kernels/gemm/igemm/igemm_warp_specialized.sm_86.cubin",
    "kernels/gemm/igemm/igemm_tiled.sm_86.cubin"
)

cat("== handtune_imma_s04: S08 -> S04 on IMMA stalls ==\n")
for (t in targets) {
    if (!file.exists(t)) next
    out <- sub("\\.sm_86\\.cubin$", ".imma_s04.sm_86.cubin", t)
    patch_imma(t, out)
}
cat("\nDone. Bench harness must be updated to load *.imma_s04.sm_86.cubin variants.\n")

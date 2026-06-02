# zzz.R -- package load hooks.

# WSL CUDA passthrough: nvidia-smi needs libnvidia-ml.so from
# /usr/lib/wsl/lib. An R subprocess inherits a stripped LD_LIBRARY_PATH
# that often misses it. Prepend it at attach time so capture_gpu_state()
# and friends can reach the GPU. No-op off WSL or when already present.
# (Previously a source-time guard at the top of scripts/bench/bench_meta.R;
# as package code it must run in .onLoad, not at build time.)
.onLoad <- function(libname, pkgname) {
  wsl_cuda_lib <- "/usr/lib/wsl/lib"
  if (dir.exists(wsl_cuda_lib) &&
      !grepl(wsl_cuda_lib, Sys.getenv("LD_LIBRARY_PATH"), fixed = TRUE)) {
    current <- Sys.getenv("LD_LIBRARY_PATH")
    Sys.setenv(LD_LIBRARY_PATH = if (nzchar(current))
                                   paste(wsl_cuda_lib, current, sep = ":")
                                 else wsl_cuda_lib)
  }
}

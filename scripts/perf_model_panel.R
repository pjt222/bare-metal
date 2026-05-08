#!/usr/bin/env Rscript
# perf_model_panel.R — Roofline + compute/memory roof panel for GA104
# Usage: Rscript scripts/perf_model_panel.R <M> <N> <K> [kernel_name]
#        or interactive

cat("=== GA104 (RTX 3070 Ti) Performance Model Panel ===\n\n")

# Hardware peaks (sm_86) — best measured from warmup runs
FP32_PEAK_TFLOPS <- 21.7          # TFLOPS (non-Tensor)
FP16_TC_PEAK_TFLOPS <- 174.0       # Tensor Core FP16
INT8_TC_PEAK_TFLOPS <- 348.0       # Tensor Core INT8 (2x FP16)
DRAM_BW_GBPS <- 608.0              # GB/s peak memory bandwidth
L2_BW_GBPS <- 3000.0               # ~3 TB/s L2 bandwidth (arch estimate)
SMEM_BW_GBPS <- 16000.0            # ~16 TB/s shared mem bandwidth

# Operational intensities (FLOP / byte) for common kernels
oi_dense_gemm <- function(M, N, K) { (2.0 * M * N * K) / (2.0 * (M*K + K*N + M*N)) }
oi_sparse_gemm <- function(M, N, K) { (2.0 * M * N * (K/2)) / (1.0*M*(K/2) + 2.0*K*N + 4.0*M*N) }
oi_flash_attn <- function(B, H, S, D) { (4.0 * B * H * S * S * D) / (4.0 * B * H * S * D) }
oi_softmax <- function(B, N) { (5.0 * B * N) / (4.0 * B * N) }
oi_layernorm <- function(B, N) { (5.0 * B * N) / (4.0 * B * N) }

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 3) {
  M <- as.numeric(args[1]); N <- as.numeric(args[2]); K <- as.numeric(args[3])
  kernel <- ifelse(length(args) >= 4, args[4], "gemm")
} else if (length(args) == 1 && args[1] == "interactive") {
  cat("Interactive mode:\n")
  M <- as.numeric(readline("M (rows): ")); N <- as.numeric(readline("N (cols): ")); K <- as.numeric(readline("K (inner): "))
  kernel <- readline("Kernel type [dense_gemm/sparse_gemm/flash_attn/softmax/layernorm]: ")
} else {
  # Default: typical LLM GEMM size
  M <- 4096; N <- 4096; K <- 4096
  kernel <- "dense_gemm"
  cat("Using defaults: M=4096 N=4096 K=4096 kernel=dense_gemm\n")
  cat("Usage: Rscript scripts/perf_model_panel.R <M> <N> <K> [kernel]\n")
}

cat(sprintf("\n--- %s: %.0f×%.0f×%.0f ---\n", kernel, M, N, K))

oi <- switch(kernel,
  "dense_gemm" = oi_dense_gemm(M,N,K),
  "sparse_gemm" = oi_sparse_gemm(M,N,K),
  "flash_attn" = oi_flash_attn(M,N,K,64),  # B,H,S,D = M,N,K,64
  "softmax" = oi_softmax(M,N),
  "layernorm" = oi_layernorm(M,N),
  oi_dense_gemm(M,N,K)
)

cat(sprintf("Operational Intensity: %.2f FLOP/byte\n", oi))

# Roofline: performance = min(peak_compute, peak_bw * oi)
roof_fp32 <- min(FP32_PEAK_TFLOPS, DRAM_BW_GBPS * oi / 1000)
roof_fp16 <- min(FP16_TC_PEAK_TFLOPS, DRAM_BW_GBPS * oi / 1000)
roof_int8 <- min(INT8_TC_PEAK_TFLOPS, DRAM_BW_GBPS * oi / 1000)

cat("\n--- Roofline Predictions (DRAM roof) ---\n")
cat(sprintf("  FP32 scalar:     %.1f TFLOPS (%.1f%% of peak)\n", roof_fp32, 100*roof_fp32/FP32_PEAK_TFLOPS))
cat(sprintf("  FP16 Tensor:     %.1f TFLOPS (%.1f%% of peak)\n", roof_fp16, 100*roof_fp16/FP16_TC_PEAK_TFLOPS))
cat(sprintf("  INT8 Tensor:     %.1f TFLOPS (%.1f%% of peak)\n", roof_int8, 100*roof_int8/INT8_TC_PEAK_TFLOPS))

# L2 roofline (kernels with good reuse)
roof_l2_fp32 <- min(FP32_PEAK_TFLOPS, L2_BW_GBPS * oi / 1000)
roof_l2_fp16 <- min(FP16_TC_PEAK_TFLOPS, L2_BW_GBPS * oi / 1000)
cat("\n--- L2-cache Roofline ---\n")
cat(sprintf("  FP32: %.1f TFLOPS | FP16: %.1f TFLOPS\n", roof_l2_fp32, roof_l2_fp16))

# Memory ceilings per level
mem_limits <- data.frame(
  level = c("DRAM", "L2", "Shared"),
  bw_gbps = c(DRAM_BW_GBPS, L2_BW_GBPS, SMEM_BW_GBPS),
  ceiling_tf = c(DRAM_BW_GBPS * oi / 1000, L2_BW_GBPS * oi / 1000, SMEM_BW_GBPS * oi / 1000)
)

cat("\n--- Memory Bandwidth Bottleneck Analysis ---\n")
for (i in seq_len(nrow(mem_limits))) {
  limited <- mem_limits$ceiling_tf[i] < INT8_TC_PEAK_TFLOPS
  cat(sprintf("  %-8s: %.0f GB/s → %.1f TFLOPS ceiling [%s]\n",
              mem_limits$level[i], mem_limits$bw_gbps[i], mem_limits$ceiling_tf[i],
              if (limited) "BOUND" else "compute-bound"))
}

# Achievable with occupancy
cat("\n--- Occupancy Effect (16 warps/block case) ---\n")
for (warps in c(8, 16, 32, 48)) {
  occ <- 100 * warps / 48
  eff <- min(1.0, warps / 32)  # rough: need 32 warps to hide latency
  cat(sprintf("  %2d warps/SM (%.0f%% occ): effective ~%.0f%% of peak\n", warps, occ, 100*eff))
}

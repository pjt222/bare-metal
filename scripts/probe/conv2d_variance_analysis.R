#!/usr/bin/env Rscript
# scripts/probe/conv2d_variance_analysis.R
#
# Analysis of the #195 campaign, results/probe/conv2d_variance.jsonl.
# Run from the repo root:  Rscript scripts/probe/conv2d_variance_analysis.R
#
# Part 1 applies the pre-registered void rules
#        (results/probe/conv2d_variance.prereg.md).
# Part 2 is the pre-registered comparison, as far as the design allowed:
#        see "Deviations" in results/probe/conv2d_variance.results.md.
# Part 3 is EXPLORATORY and was not pre-registered.
# These three blocks are the exact code that produced the results file.

# ---- Part 1: void rules (pre-registered) ------------------------------------
suppressMessages(library(jsonlite))
d <- stream_in(file("results/probe/conv2d_variance.jsonl"), verbose = FALSE)
cat("rows:", nrow(d), " samples:", length(unique(d$sample_idx)), " run_id:", unique(d$run_id), " git:", unique(d$git_head), "\n")
cat("columns present:", paste(intersect(c("burnin","burnin_n","block","pre_snapshot","post_snapshot","pre_power_below_max","pre_load_1m"), names(d)), collapse=", "), "\n\n")
s <- d[!duplicated(d$sample_idx), ]
ok_thr <- function(x) is.na(x) | x %in% c("GpuIdle", "")
s$void_power <- is.na(s$post_power_limit_w) | s$post_power_limit_w != 150
s$void_thr   <- !(ok_thr(s$pre_throttle) & ok_thr(s$post_throttle))
s$void_clk   <- is.na(s$post_clock_sm) | s$post_clock_sm < 1600
s$void <- s$void_power | s$void_thr | s$void_clk
show <- s[, c("sample_idx","condition","burnin","pre_pstate","pre_clock_sm","post_clock_sm","post_power_limit_w","pre_throttle","post_throttle","pre_load_1m","post_load_1m","post_temp_c","void")]
print(show, row.names = FALSE)
ab <- s[!s$burnin, ]
cat("\nVOID among A/B samples:", sum(ab$void), "of", nrow(ab), "  (rule: >= 2 voids the run)\n")
cat("  by reason: power", sum(ab$void_power), " throttle", sum(ab$void_thr), " clock", sum(ab$void_clk), "\n")
cat("NA throttles (unreadable mask):", sum(is.na(s$pre_throttle) | is.na(s$post_throttle)), "\n")
cat("power refs NA anywhere:", anyNA(d$pre_power_below_max) || anyNA(d$post_power_below_max), "; snapshot missing anywhere:", any(!d$pre_snapshot | !d$post_snapshot), "\n")

# ---- Part 2: pre-registered comparison --------------------------------------
suppressMessages(library(jsonlite))
d <- stream_in(file("results/probe/conv2d_variance.jsonl"), verbose = FALSE)
ab <- d[!d$burnin, ]
meas <- c("sd64_explicit","sd64_implicit","sd32_explicit","sd32_implicit","sd128_explicit","sd128_implicit")
cat("== 1. Per measurement x condition (A = back-to-back, entered P0; B = 90 s idle, entered P8)\n")
cat(sprintf("%-15s %3s | %-6s %7s %6s %6s %7s %7s %6s\n","measurement","ord","cond","mean","sd","rel%","min","max","max/min"))
for (m in meas) for (cd in c("A","B")) {
  x <- ab$throughput[ab$measurement == m & ab$condition == cd]
  cat(sprintf("%-15s %3d | %-6s %7.0f %6.0f %6.2f %7.0f %7.0f %6.3f\n", m, ab$order_in_run[ab$measurement==m][1], cd, mean(x), sd(x), 100*sd(x)/mean(x), min(x), max(x), max(x)/min(x)))
}
cat("\n== 2. Condition (= entry state) effect per measurement, Welch t-test, A vs B\n")
for (m in meas) {
  a <- ab$throughput[ab$measurement==m & ab$condition=="A"]; b <- ab$throughput[ab$measurement==m & ab$condition=="B"]
  t <- t.test(a, b)
  cat(sprintf("%-15s A-B = %+6.0f GFLOPS (%+5.1f%% of B)  95%% CI [%+.0f, %+.0f]  p = %.2g\n", m, mean(a)-mean(b), 100*(mean(a)-mean(b))/mean(b), t$conf.int[1], t$conf.int[2], t$p.value))
}
cat("\n== 3. Pooled across conditions: run-to-run span per measurement (all 30 A/B samples)\n")
for (m in meas) { x <- ab$throughput[ab$measurement==m]; cat(sprintf("%-15s mean %6.0f  rel sd %5.2f%%  max/min %.3f\n", m, mean(x), 100*sd(x)/mean(x), max(x)/min(x))) }
cat("\n== 4. Every sample, sd64_implicit and sd64_explicit (row by row)\n")
w <- reshape(d[d$measurement %in% c("sd64_explicit","sd64_implicit"), c("sample_idx","condition","measurement","throughput")], idvar=c("sample_idx","condition"), timevar="measurement", direction="wide")
names(w) <- sub("throughput.","",names(w)); print(w[order(w$sample_idx),], row.names=FALSE)
cat("\n== 5. Lead-in: W samples vs first A and first B, all six measurements\n")
for (m in meas) { x <- d[d$measurement==m,]; cat(sprintf("%-15s W1 %6.0f  W2 %6.0f | A3 %6.0f  B4 %6.0f | A-mean %6.0f  B-mean %6.0f\n", m, x$throughput[x$sample_idx==1], x$throughput[x$sample_idx==2], x$throughput[x$sample_idx==3], x$throughput[x$sample_idx==4], mean(x$throughput[x$condition=="A"]), mean(x$throughput[x$condition=="B"]))) }
cat("\n== 6. Host load as covariate: within-condition correlation of throughput with pre_load_1m\n")
for (m in meas) { r <- sapply(c("A","B"), function(cd) { x <- ab[ab$measurement==m & ab$condition==cd,]; cor(x$throughput, x$pre_load_1m) }); cat(sprintf("%-15s r(A) = %+.2f  r(B) = %+.2f\n", m, r[1], r[2])) }
cat("\n== 7. Drift: within-condition correlation of throughput with sample_idx\n")
for (m in meas) { r <- sapply(c("A","B"), function(cd) { x <- ab[ab$measurement==m & ab$condition==cd,]; cor(x$throughput, x$sample_idx) }); cat(sprintf("%-15s r(A) = %+.2f  r(B) = %+.2f\n", m, r[1], r[2])) }

# ---- Part 3: EXPLORATORY (not pre-registered) -------------------------------
# EXPLORATORY -- not pre-registered. Labelled as such wherever reported.
suppressMessages(library(jsonlite))
d <- stream_in(file("results/probe/conv2d_variance.jsonl"), verbose = FALSE)
ab <- d[!d$burnin, ]
w <- reshape(ab[, c("sample_idx","condition","measurement","throughput")], idvar=c("sample_idx","condition"), timevar="measurement", direction="wide")
names(w) <- sub("throughput.","",names(w))
s <- ab[ab$measurement=="sd64_implicit", c("sample_idx","post_clock_sm","post_temp_c","pre_temp_c","post_power_w","pre_load_1m","post_clock_mem")]
w <- merge(w, s, by="sample_idx")
x <- w$sd64_implicit
cat("sd64_implicit sorted:\n"); print(sort(x))
gap <- diff(sort(x)); cat(sprintf("largest gap in sorted values: %.0f GFLOPS between %.0f and %.0f\n", max(gap), sort(x)[which.max(gap)], sort(x)[which.max(gap)+1]))
thr <- (sort(x)[which.max(gap)] + sort(x)[which.max(gap)+1]) / 2
w$cluster <- ifelse(x > thr, "high", "low")
cat(sprintf("split at %.0f: high n=%d (mean %.0f), low n=%d (mean %.0f), ratio %.3f\n", thr, sum(w$cluster=="high"), mean(x[w$cluster=="high"]), sum(w$cluster=="low"), mean(x[w$cluster=="low"]), mean(x[w$cluster=="high"])/mean(x[w$cluster=="low"])))
cat("\ncluster by condition:\n"); print(table(w$condition, w$cluster))
cat("\nmeans by cluster of other variables in the SAME sample:\n")
for (v in c("sd64_explicit","sd32_explicit","sd32_implicit","sd128_explicit","sd128_implicit","post_clock_sm","post_temp_c","pre_temp_c","pre_load_1m","post_power_w")) cat(sprintf("  %-15s high %8.1f  low %8.1f\n", v, mean(w[[v]][w$cluster=="high"]), mean(w[[v]][w$cluster=="low"])))
cat("\ncluster sequence by sample_idx (H/L):\n"); w <- w[order(w$sample_idx),]; cat(paste0(substr(toupper(w$cluster),1,1), collapse=""), "\n")
cat("runs of the same cluster:", length(rle(w$cluster)$lengths), "of", nrow(w), "samples\n")

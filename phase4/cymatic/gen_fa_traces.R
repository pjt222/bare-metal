#!/usr/bin/env Rscript
# gen_fa_traces.R
#
# Variant of gen_cymatic_data.R that injects Flash Attention block-level
# access traces in addition to (or in place of) the default trace set.
# Used by Issue #94 to measure whether cymatic helps real FA workloads.
#
# Usage:
#   Rscript gen_fa_traces.R [grid_n=1024] [n=6] [m=4]
#
# FA configuration assumed: Br=16 (queries/tile), Bc=64 (keys/tile).
# At seq=N: Nq = N/Br, Nk = N/Bc query/key tiles. Each FA iteration is
# one (q_tile, k_tile) pair. Trace = sequence of those pairs.
#
# We map (q_tile_idx, k_tile_idx) -> grid cell (i, j). The grid must be
# at least max(Nq, Nk) on each axis. Run from phase4/cymatic/.

source("../../scripts/cymatic/cymatic_mapping.R")
source("../../scripts/cymatic/cymatic_analyze.R")

args   <- commandArgs(trailingOnly = TRUE)
grid_n <- if (length(args) >= 1) as.integer(args[1]) else 1024L
n1     <- if (length(args) >= 2) as.integer(args[2]) else 6L
m1     <- if (length(args) >= 3) as.integer(args[3]) else 4L

cat(sprintf("[gen_fa] grid=%d  mode=(%d, %d)\n", grid_n, n1, m1))
mapping <- cymatic_mapping(grid_n = grid_n, modes = list(list(n=n1, m=m1, alpha=1.0)))

# ---- Row-major-inside ordering --------------------------------------------
inside       <- mapping$grid$inside
inside_cells <- which(inside, arr.ind = TRUE)
ord          <- order(inside_cells[, 1], inside_cells[, 2])
inside_cells <- inside_cells[ord, , drop = FALSE]
n_inside     <- nrow(inside_cells)

ij_key     <- paste(inside_cells[, 1], inside_cells[, 2], sep = "_")
rmi_lookup <- setNames(seq_len(n_inside) - 1L, ij_key)

ij_to_rmi <- function(cells) {
    k   <- paste(cells[, 1], cells[, 2], sep = "_")
    out <- rmi_lookup[k]
    out[!is.na(out)]
}

perm <- mapping$address[inside_cells] - 1L
stopifnot(min(perm) == 0L, max(perm) == n_inside - 1L)
stopifnot(length(unique(perm)) == n_inside)

con <- file("perm.bin", "wb")
writeBin(as.integer(grid_n),   con, size = 4)
writeBin(as.integer(n_inside), con, size = 4)
writeBin(as.integer(perm),     con, size = 4)
close(con)
cat(sprintf("[gen_fa] wrote perm.bin  n_inside=%d  buffer=%.1f MB\n",
            n_inside, n_inside * 4 / 1e6))

# ---- FA trace generators --------------------------------------------------

# Translate trace cells to grid coordinates with origin at center.
center_offset <- function(cells, grid_n) {
    cells[, 1] <- cells[, 1] + as.integer(grid_n / 2)
    cells[, 2] <- cells[, 2] + as.integer(grid_n / 2)
    cells
}

# Block-level FA: each iteration is one (q_block_idx, k_block_idx) pair.
# Per query block, scan all key blocks (causal mask off; full attention).
# Each (q, k) pair is mapped to D_HEAD adjacent cells in the row direction
# to model the per-element access within one block.
fa_block_trace <- function(seq_len, Br, Bc, D_head, grid_n,
                           order = c("rowmajor", "diagonal", "zigzag")) {
    order <- match.arg(order)
    Nq <- as.integer(seq_len / Br)
    Nk <- as.integer(seq_len / Bc)

    pairs <- switch(order,
        rowmajor = {
            ix <- expand.grid(q = 0:(Nq - 1), k = 0:(Nk - 1))
            ix[order(ix$q, ix$k), , drop = FALSE]
        },
        diagonal = {
            ix <- expand.grid(q = 0:(Nq - 1), k = 0:(Nk - 1))
            ix[order((ix$q + ix$k), ix$q), , drop = FALSE]
        },
        zigzag = {
            rows <- lapply(0:(Nq - 1), function(q) {
                ks <- if (q %% 2 == 0) 0:(Nk - 1) else (Nk - 1):0
                data.frame(q = q, k = ks)
            })
            do.call(rbind, rows)
        }
    )

    # For each (q, k) pair, the kernel touches Bc * D_head elements of K.
    # Model as a sequence of D_head cells at row k along the j axis,
    # repeated for each row in the K block.
    cells <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(idx) {
        q <- pairs$q[idx]; k <- pairs$k[idx]
        ki <- (k * Bc + seq.int(0, Bc - 1)) %% grid_n
        kj <- seq.int(0, D_head - 1L)
        cbind(rep(ki, each = D_head), rep(kj, times = Bc))
    }))
    storage.mode(cells) <- "integer"
    # Shift to 1-indexed within (1..grid_n)
    cells <- cells + 1L
    cells
}

# Configurations to test. seq=1024, Br=16, Bc=64, D=64 -> Nq=64, Nk=16.
# That fits comfortably in a 1024-grid (block addresses are 0..1023).
seqs   <- c(512L, 1024L)
orders <- c("rowmajor", "diagonal", "zigzag")
Br <- 16L; Bc <- 64L; D_head <- 64L

build_trace <- function(name, cells) {
    rmi <- ij_to_rmi(cells)
    list(name = name, rmi = as.integer(rmi))
}

traces <- list()
for (s in seqs) for (ord in orders) {
    cells <- fa_block_trace(s, Br, Bc, D_head, grid_n, order = ord)
    nm <- sprintf("fa_seq%d_%s", s, ord)
    tr <- build_trace(nm, cells)
    traces[[length(traces) + 1L]] <- tr
    cat(sprintf("  %-30s n=%d cells (raw=%d, in-disc kept)\n",
                nm, length(tr$rmi), nrow(cells)))
}

# Add the rowmajor_full control trace for direct comparison
traces[[length(traces) + 1L]] <- build_trace("rowmajor_full", inside_cells)
cat(sprintf("  %-30s n=%d cells (control)\n", "rowmajor_full", n_inside))

# ---- Write traces.bin -----------------------------------------------------
con <- file("traces.bin", "wb")
writeBin(as.integer(length(traces)), con, size = 4)
for (tr in traces) {
    name_b <- charToRaw(tr$name); name_b <- c(name_b, as.raw(0))
    writeBin(as.integer(length(name_b)), con, size = 4)
    writeBin(name_b, con)
    writeBin(as.integer(length(tr$rmi)), con, size = 4)
    writeBin(as.integer(tr$rmi), con, size = 4)
}
close(con)
cat(sprintf("[gen_fa] wrote traces.bin (%d FA + 1 control trace)\n",
            length(traces) - 1L))

NVCC      ?= nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -arch=$(ARCH) -O2 -std=c++17

.PHONY: all gen run clean

all: bench

bench: bench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Generate input files (perm.bin + traces.bin) using R.
# Default: 1024-grid, mode (6, 4). Override with GRID=, N=, M=.
GRID ?= 1024
N    ?= 6
M    ?= 4

gen:
	Rscript gen_cymatic_data.R $(GRID) $(N) $(M)

run: bench perm.bin traces.bin
	./bench

# Sweep grid sizes (L1 / L2 / DRAM regimes). Captures per-grid results
# into the centralised results/ tree (audit Tier 7).
RESULTS_DIR ?= ../../results/cymatic/grids
sweep: bench
	@mkdir -p $(RESULTS_DIR)
	@for g in 256 512 1024 2048 ; do \
		echo "=== GRID=$$g ==="; \
		Rscript gen_cymatic_data.R $$g $(N) $(M) > /dev/null 2>&1; \
		./bench 200 5 11 | tee $(RESULTS_DIR)/grid$${g}_results.txt; \
		echo ""; \
	done

clean:
	rm -f bench perm.bin traces.bin mapping.rds cymatic_mapping.csv

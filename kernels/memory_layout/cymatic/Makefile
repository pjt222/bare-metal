NVCC      ?= nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -arch=$(ARCH) -O2 -std=c++17

.PHONY: all gen run clean

all: bench

bench: bench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Generate input files (perm.bin + traces.bin) using R.
# Default: 1024-grid, mode (6, 4), original disc domain. Override with GRID=, N=, M=, DOMAIN=disc|square|overlayed.
GRID ?= 1024
N    ?= 6
M    ?= 4
DOMAIN ?= disc

gen:
	Rscript gen_cymatic_data.R $(GRID) $(N) $(M) 0 0 0 $(DOMAIN)

run: bench perm.bin traces.bin
	./bench 200 5 11 perm_$(DOMAIN).bin traces_$(DOMAIN).bin

# Sweep grid sizes (L1 / L2 / DRAM regimes). Captures per-grid results
# into the centralised results/ tree.
RESULTS_DIR ?= ../../../results/cymatic/grids
sweep: bench
	@mkdir -p $(RESULTS_DIR)
	@for g in 256 512 1024 2048 ; do \
		echo "=== GRID=$$g ==="; \
		Rscript gen_cymatic_data.R $$g $(N) $(M) 0 0 0 $(DOMAIN) > /dev/null 2>&1; \
		./bench 200 5 11 perm_$(DOMAIN).bin traces_$(DOMAIN).bin | tee $(RESULTS_DIR)/grid$${g}_$(DOMAIN)_results.txt; \
		echo ""; \
	done

clean:
	rm -f bench perm.bin traces.bin mapping.rds cymatic_mapping.csv

NVCC      ?= nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -arch=$(ARCH) -O2 -std=c++17

.PHONY: all gen run clean

all: bench_cymatic

bench_cymatic: bench_cymatic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Generate input files (perm.bin + traces.bin) using R.
# Default: 1024-grid, mode (6, 4). Override with GRID=, N=, M=.
GRID ?= 1024
N    ?= 6
M    ?= 4

gen:
	Rscript gen_cymatic_data.R $(GRID) $(N) $(M)

run: bench_cymatic perm.bin traces.bin
	./bench_cymatic

# Sweep grid sizes (L1 / L2 / DRAM regimes). Captures per-grid results.
sweep: bench_cymatic
	@mkdir -p results
	@for g in 256 512 1024 2048 ; do \
		echo "=== GRID=$$g ==="; \
		Rscript gen_cymatic_data.R $$g $(N) $(M) > /dev/null 2>&1; \
		./bench_cymatic 200 5 11 | tee results/grid$${g}_results.txt; \
		echo ""; \
	done

clean:
	rm -f bench_cymatic perm.bin traces.bin mapping.rds cymatic_mapping.csv

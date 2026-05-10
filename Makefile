# bare-metal GPU — Top-level Makefile
#
# Usage:
#   make reproduce    — setup + verify + build + bench (full one-stop)
#   make setup        — renv::restore() + install local cuasmR R package
#   make verify       — environment check (CUDA, GPU, cuasmR)
#   make all          — build all kernel cubins + benchmark executables
#   make bench        — run benches vs docs/baselines.json
#   make cubins       — compile all .cu files to .cubin
#   make benches      — compile all bench*.cu files to executables
#   make phaseN       — build phase N only (N=1..5)
#   make test         — run all benchmark executables (smoke test)
#   make clean        — remove all generated artifacts
#   make disasm       — disassemble all cubins to .sass

SM_ARCH     := sm_86
NVCC        := nvcc
NVCC_FLAGS  := -arch=$(SM_ARCH) -O2
CUBIN_FLAGS := --cubin -arch=$(SM_ARCH) -O2

RSCRIPT     := Rscript

# ------------------------------------------------------------------
# Find all source files
# ------------------------------------------------------------------
# experiments/ has its own build conventions (Driver-API harness, Rust
# kernels). Excluded from the default Makefile sweep.
KERNEL_CU   := $(shell find . \( -path ./tools -o -path ./experiments -o -path ./renv -o -path ./.git \) -prune -o -name '*.cu' -print | grep -v 'bench' | grep -v 'host' | grep -v '^./tests')
BENCH_CU    := $(shell find . \( -path ./tools -o -path ./experiments -o -path ./renv -o -path ./.git \) -prune -o -name 'bench.cu' -print) $(shell find phase3/flash_attention -name 'bench_*.cu' -print) $(shell find phase4 -name 'bench_*.cu' -print)

KERNEL_CUBINS := $(KERNEL_CU:.cu=.$(SM_ARCH).cubin)
BENCH_EXES    := $(BENCH_CU:.cu=)

# Phase-specific bench executables
PHASE1_BENCH  :=  # tutorial — no benches, just vector_add hello-world
PHASE2_BENCH  := $(shell find phase2 -name 'bench.cu' | sed 's/\.cu//')
PHASE3_BENCH  := $(shell find phase3 -name 'bench*.cu' | sed 's/\.cu//')
PHASE4_BENCH  := $(shell find phase4 -name 'bench*.cu' | sed 's/\.cu//')
PHASE5_BENCH  := $(shell find phase5 -name 'bench*.cu' | sed 's/\.cu//')

# ------------------------------------------------------------------
# Default target
# ------------------------------------------------------------------
.PHONY: all cubins benches phase1 phase2 phase3 phase4 phase5 test clean disasm help \
        setup verify bench reproduce \
        tutorial gemm reductions attention convolution elementwise memory_layout composition

all: cubins benches

cubins: $(KERNEL_CUBINS)

benches: $(BENCH_EXES)

# ------------------------------------------------------------------
# Generic rules
# ------------------------------------------------------------------
%.$(SM_ARCH).cubin: %.cu
	@echo "[CUBIN] $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(CUBIN_FLAGS) -o $@ $<

# Benchmarks in phase2+ need common headers and -lcuda
phase2/%/bench: phase2/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Iphase2/common

phase3/%/bench: phase3/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Iphase2/common

phase4/%/bench: phase4/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Iphase2/common

phase5/%/bench: phase5/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Iphase2/common

kernels/tutorial/host: kernels/tutorial/host.cu
	@echo "[HOST]  $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

# Flash Attention has multiple bench variants
phase3/flash_attention/bench_%: phase3/flash_attention/bench_%.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Iphase2/common

# ------------------------------------------------------------------
# Phase targets
# ------------------------------------------------------------------
phase1 tutorial: kernels/tutorial/vector_add.$(SM_ARCH).cubin kernels/tutorial/host

phase2: $(shell find phase2 -name '*.cu' ! -name 'bench*.cu' | sed 's/\.cu/.$(SM_ARCH).cubin/') $(PHASE2_BENCH)

phase3: $(shell find phase3 -name '*.cu' ! -name 'bench*.cu' | sed 's/\.cu/.$(SM_ARCH).cubin/') $(PHASE3_BENCH)

phase4: $(shell find phase4 -name '*.cu' ! -name 'bench*.cu' | sed 's/\.cu/.$(SM_ARCH).cubin/') $(PHASE4_BENCH)

phase5: $(shell find phase5 -name '*.cu' ! -name 'bench*.cu' | sed 's/\.cu/.$(SM_ARCH).cubin/') $(PHASE5_BENCH)

# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
test: $(PHASE2_BENCH)
	@echo "=== Running smoke tests ==="
	@for exe in $(PHASE2_BENCH); do \
		if [ -f "$$exe" ]; then \
			echo "--- $$exe ---"; \
			"$$exe" 512 512 512 || true; \
		fi; \
	done
	@echo "=== Smoke tests complete ==="

# ------------------------------------------------------------------
# Disassembly
# ------------------------------------------------------------------
disasm: cubins
	@echo "=== Disassembling cubins ==="
	@for cubin in $(KERNEL_CUBINS); do \
		if [ -f "$$cubin" ]; then \
			$(RSCRIPT) scripts/build.R disasm $$cubin >/dev/null 2>&1 || true; \
		fi; \
	done
	@echo "=== Disassembly complete ==="

# ------------------------------------------------------------------
# Reproducibility entry points (Tier 11)
#
# Single chain from a fresh clone to a passing regression check.
# 'make reproduce' executes setup -> verify -> all -> bench in order;
# any failure short-circuits the chain.
# ------------------------------------------------------------------
setup:
	@echo "=== Restoring R deps via renv ==="
	@$(RSCRIPT) -e 'if (!requireNamespace("renv", quietly=TRUE)) install.packages("renv", repos="https://cloud.r-project.org"); renv::restore()'
	@echo "=== Installing local cuasmR R package ==="
	@$(RSCRIPT) scripts/install_cuasmR.R

verify:
	@$(RSCRIPT) scripts/verify_setup.R

bench:
	@$(RSCRIPT) scripts/bench/bench_regress.R

reproduce: setup verify all bench
	@echo ""
	@echo "================================================================"
	@echo "✓ Full reproduction complete."
	@echo "  setup    — R deps installed via renv"
	@echo "  verify   — toolchain + GPU detected"
	@echo "  all      — every cubin + bench compiled"
	@echo "  bench    — results compared to docs/baselines.json"
	@echo "================================================================"

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
clean:
	@echo "Cleaning generated files..."
	@find . -path ./tools -prune -o -name '*.$(SM_ARCH).cubin' -exec rm -f {} +
	@find . -path ./tools -prune -o -name '*.cuasm' -exec rm -f {} +
	@find . -path ./tools -prune -o -name '*.sass' -exec rm -f {} +
	@find . -path ./tools -prune -o -name '*.reassembled.cubin' -exec rm -f {} +
	@for exe in $(BENCH_EXES); do rm -f $$exe 2>/dev/null || true; done
	@rm -f kernels/tutorial/host 2>/dev/null || true
	@echo "Done."

# ------------------------------------------------------------------
# Help
# ------------------------------------------------------------------
help:
	@echo "bare-metal GPU build system"
	@echo ""
	@echo "Reproducibility (Tier 11):"
	@echo "  make reproduce — one-stop: setup + verify + build + bench"
	@echo "  make setup     — renv::restore() + install cuasmR"
	@echo "  make verify    — environment check (CUDA, GPU, cuasmR)"
	@echo "  make bench     — run benches vs docs/baselines.json"
	@echo ""
	@echo "Build:"
	@echo "  make all       — build all cubins + benches"
	@echo "  make cubins    — compile all .cu files to .cubin"
	@echo "  make benches   — compile all bench*.cu to executables"
	@echo "  make phaseN    — build phase N only (N=1..5)"
	@echo "  make test      — run smoke tests on compiled benches"
	@echo "  make disasm    — disassemble all cubins to .sass"
	@echo "  make clean     — remove all generated artifacts"
	@echo "  make help      — show this message"

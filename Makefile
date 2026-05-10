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
BENCH_CU    := $(shell find . \( -path ./tools -o -path ./experiments -o -path ./renv -o -path ./.git \) -prune -o -name 'bench.cu' -print) $(shell find kernels -name 'bench_*.cu' -print 2>/dev/null)

KERNEL_CUBINS := $(KERNEL_CU:.cu=.$(SM_ARCH).cubin)
BENCH_EXES    := $(BENCH_CU:.cu=)

# Family-specific bench executables (Tier 13 reorg replaces phase{1..5}/).
GEMM_BENCH         := $(shell find kernels/gemm          -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
REDUCTIONS_BENCH   := $(shell find kernels/reductions    -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
ELEMENTWISE_BENCH  := $(shell find kernels/elementwise   -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
ATTENTION_BENCH    := $(shell find kernels/attention     -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
CONVOLUTION_BENCH  := $(shell find kernels/convolution   -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
MEMORY_LAYOUT_BENCH:= $(shell find kernels/memory_layout -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
COMPOSITION_BENCH  := $(shell find kernels/composition   -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')

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

# Generic bench rule for any kernels/<family>/<kernel>/bench.cu. Per-dir
# multi-variant rules (bench_*.cu) follow below; make's % can't bind two
# different segments in one target so we list them family-by-family.
kernels/%/bench: kernels/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

kernels/tutorial/host: kernels/tutorial/host.cu
	@echo "[HOST]  $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

# Flash Attention has multiple bench variants
kernels/attention/flash_attention/bench_%: kernels/attention/flash_attention/bench_%.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

# Conv2d / resblock have multiple bench variants too
kernels/convolution/conv2d/bench_%: kernels/convolution/conv2d/bench_%.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

kernels/convolution/resblock/bench_%: kernels/convolution/resblock/bench_%.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

kernels/gemm/hgemm/bench_%: kernels/gemm/hgemm/bench_%.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

kernels/gemm/igemm/bench_%: kernels/gemm/igemm/bench_%.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

# ------------------------------------------------------------------
# Phase targets
# ------------------------------------------------------------------
# Family targets: build all cubins + benches under kernels/<family>/.
tutorial: kernels/tutorial/vector_add.$(SM_ARCH).cubin kernels/tutorial/host

gemm:           $(shell find kernels/gemm          -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(GEMM_BENCH)
reductions:     $(shell find kernels/reductions    -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(REDUCTIONS_BENCH)
elementwise:    $(shell find kernels/elementwise   -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(ELEMENTWISE_BENCH)
attention:      $(shell find kernels/attention     -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(ATTENTION_BENCH)
convolution:    $(shell find kernels/convolution   -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(CONVOLUTION_BENCH)
memory_layout:  $(shell find kernels/memory_layout -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(MEMORY_LAYOUT_BENCH)
composition:    $(shell find kernels/composition   -name '*.cu' ! -name 'bench*.cu' 2>/dev/null | sed 's/\.cu/.$(SM_ARCH).cubin/') $(COMPOSITION_BENCH)

# Legacy phaseN aliases — forward to family targets so old invocations
# (e.g. tutorials, scripts) keep working post-Tier-13.
phase1: tutorial
phase2: gemm reductions elementwise
phase3: attention
phase4: attention convolution reductions elementwise memory_layout
phase5: composition

# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
# Smoke test: run all GEMM + reduction + elementwise benches at 512^3.
test: $(GEMM_BENCH) $(REDUCTIONS_BENCH) $(ELEMENTWISE_BENCH)
	@echo "=== Running smoke tests ==="
	@for exe in $(GEMM_BENCH) $(REDUCTIONS_BENCH) $(ELEMENTWISE_BENCH); do \
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
	@echo "  make test      — run smoke tests on compiled benches"
	@echo "  make disasm    — disassemble all cubins to .sass"
	@echo "  make clean     — remove all generated artifacts"
	@echo ""
	@echo "By family (Tier 13):"
	@echo "  make tutorial      — vector_add hello-world"
	@echo "  make gemm          — sgemm/hgemm/hgemm_sparse/igemm"
	@echo "  make reductions    — softmax/layernorm/groupnorm"
	@echo "  make elementwise   — activations/timestep_emb"
	@echo "  make attention     — flash_attention/cross_attention"
	@echo "  make convolution   — conv2d/resblock"
	@echo "  make memory_layout — cymatic"
	@echo "  make composition   — attention_layer"
	@echo "  make phaseN        — alias → corresponding family targets"
	@echo ""
	@echo "  make help      — show this message"

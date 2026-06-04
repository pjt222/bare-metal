# bare-metal GPU — Top-level Makefile
#
# Usage:
#   make reproduce    — setup + verify + build + bench + figures (full one-stop)
#   make setup        — renv::restore() + install local cuasmR R package
#   make verify       — environment check (CUDA, GPU, cuasmR)
#   make all          — build all kernel cubins + benchmark executables
#   make bench        — run benches vs data/baselines.json
#   make cubins       — compile all .cu files to .cubin
#   make benches      — compile all bench*.cu files to executables

#   make test         — run all benchmark executables (smoke test)
#   make clean        — remove all generated artifacts
#   make disasm       — disassemble all cubins to .sass

SM_ARCH     := sm_86
NVCC        := nvcc
NVCC_FLAGS  := -arch=$(SM_ARCH) -O2
CUBIN_FLAGS := --cubin -arch=$(SM_ARCH) -O2
REFERENCE_NVCC_FLAGS := $(NVCC_FLAGS) -std=c++17

RSCRIPT     := Rscript

CUSPARSELT_HOME ?= $(HOME)/.local/cusparselt
REFERENCE_INC_FLAGS := -Ikernels/reference/_common -I/usr/include -I/usr/include/x86_64-linux-gnu
REFERENCE_LIB_FLAGS := -lcudart -lcublas -lcublasLt -lcudnn
ifneq ($(wildcard $(CUSPARSELT_HOME)/include/cusparseLt.h),)
REFERENCE_INC_FLAGS += -I$(CUSPARSELT_HOME)/include
REFERENCE_LIB_FLAGS += -L$(CUSPARSELT_HOME)/lib64 -Xlinker -rpath -Xlinker $(CUSPARSELT_HOME)/lib64 -lcusparseLt
endif

# ------------------------------------------------------------------
# Find all source files
# ------------------------------------------------------------------
# experiments/ has its own build conventions (Driver-API harness, Rust
# kernels). Excluded from the default Makefile sweep.
KERNEL_CU   := $(shell find . \( -path ./tools -o -path ./experiments -o -path ./renv -o -path ./.git \) -prune -o -name '*.cu' -print | grep -v 'bench' | grep -v 'host' | grep -v '^./tests')
BENCH_CU    := $(shell find . \( -path ./tools -o -path ./experiments -o -path ./renv -o -path ./.git \) -prune -o -name 'bench.cu' -print) $(shell find kernels -name 'bench_*.cu' -print 2>/dev/null)

KERNEL_CUBINS := $(KERNEL_CU:.cu=.$(SM_ARCH).cubin)
BENCH_EXES    := $(BENCH_CU:.cu=)

# Family-specific bench executables.
GEMM_BENCH         := $(shell find kernels/gemm          -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
REDUCTIONS_BENCH   := $(shell find kernels/reductions    -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
ELEMENTWISE_BENCH  := $(shell find kernels/elementwise   -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
ATTENTION_BENCH    := $(shell find kernels/attention     -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
CONVOLUTION_BENCH  := $(shell find kernels/convolution   -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
MEMORY_LAYOUT_BENCH:= $(shell find kernels/memory_layout -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
COMPOSITION_BENCH  := $(shell find kernels/composition   -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')
REFERENCE_BENCH    := $(shell find kernels/reference     -name 'bench*.cu' 2>/dev/null | sed 's/\.cu//')

# Regression-baselined attention/convolution benches. data/baselines.json
# carries baselines for these two kernels, but their exes live outside the
# GEMM/reductions/elementwise groups `make test` builds — so the pre-push
# bench_regress.R SKIPped attention + convolution coverage. Listed
# explicitly (not the whole ATTENTION_BENCH / CONVOLUTION_BENCH families)
# to keep `make test` scope at exactly what regression coverage needs.
REGRESS_BENCH := \
  kernels/attention/flash_attention/bench_br16_regpv \
  kernels/convolution/conv2d/bench_implicit_gemm

# ------------------------------------------------------------------
# Default target
# ------------------------------------------------------------------
.PHONY: all cubins benches test clean disasm sass help \
        setup verify bench bench-all bench-reference compare-reference reference-pipeline reproduce figures \
        publish-hf \
        tutorial gemm reductions attention convolution elementwise memory_layout composition reference

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

# Generic bench rule for kernels/<family>/<kernel>/bench.cu.
kernels/%/bench: kernels/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda -Ikernels/_common

kernels/tutorial/host: kernels/tutorial/host.cu
	@echo "[HOST]  $<"
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

# Per-directory bench_*.cu variants. Make's % can only bind a single
# stem, so we generate one rule per directory via $(eval). To add a new
# directory with bench variants, append it to BENCH_VARIANT_DIRS.
BENCH_VARIANT_DIRS := \
  kernels/attention/cross_attention \
  kernels/attention/flash_attention \
  kernels/convolution/conv2d \
  kernels/convolution/resblock \
  kernels/gemm/hgemm \
  kernels/gemm/igemm

define BENCH_VARIANT_RULE
$(1)/bench_%: $(1)/bench_%.cu
	@echo "[BENCH] $$<"
	$$(NVCC) $$(NVCC_FLAGS) -o $$@ $$< -lcuda -Ikernels/_common
endef

$(foreach d,$(BENCH_VARIANT_DIRS),$(eval $(call BENCH_VARIANT_RULE,$(d))))

kernels/reference/%/bench: kernels/reference/%/bench.cu
	@echo "[BENCH] $<"
	$(NVCC) $(REFERENCE_NVCC_FLAGS) -o $@ $< $(REFERENCE_INC_FLAGS) $(REFERENCE_LIB_FLAGS)

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
reference:      $(REFERENCE_BENCH)

# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
# Smoke test: run all GEMM + reduction + elementwise benches at 512^3.
# Depends on `cubins` so the benches have kernels to load — the benches
# resolve their cubins at runtime via cuModuleLoad, so a bench executable
# built without its cubin runs hollow ("No kernels found"). This also
# leaves the cubins in place for scripts/bench/bench_regress.R, which the
# pre-push hook runs straight after `make test`.
#
# Each bench resolves its cubin by a path relative to cwd (e.g.
# cuModuleLoad("hgemm.sm_86.cubin")), and the cubin sits beside the bench
# source. Running the exe from the repo root therefore loads no kernels.
# Run each bench from its own directory in a subshell so the cubin
# resolves and the loop's cwd is unaffected (#130).
#
# $(REGRESS_BENCH) is a dependency but not in the smoke-run loop: those
# benches take kernel-specific args (not N N N), so running them here
# would add noise. They only need to be *built* for bench_regress.R.
test: cubins $(GEMM_BENCH) $(REDUCTIONS_BENCH) $(ELEMENTWISE_BENCH) $(REGRESS_BENCH)
	@echo "=== Running smoke tests ==="
	@for exe in $(GEMM_BENCH) $(REDUCTIONS_BENCH) $(ELEMENTWISE_BENCH); do \
		if [ -f "$$exe" ]; then \
			echo "--- $$exe ---"; \
			( cd "$$(dirname "$$exe")" && "./$$(basename "$$exe")" 512 512 512 ) || true; \
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

# Fast SASS dump: cuobjdump -sass per cubin, no R/cuasmR/renv overhead.
# `make disasm` spawns an R process per cubin (renv activation ~15s
# each) to also produce the .cuasm hand-edit format; `make sass` skips
# all that and only writes the raw .sass. Used by scripts/publish_hf.R.
# For the .cuasm hand-edit workflow, use `make disasm` or
# `Rscript scripts/build.R disasm <cubin>` directly.
sass: cubins
	@echo "=== Dumping SASS (cuobjdump) ==="
	@for cubin in $(KERNEL_CUBINS); do \
		if [ -f "$$cubin" ]; then \
			cuobjdump -sass "$$cubin" > "$${cubin%.cubin}.sass" 2>/dev/null || true; \
		fi; \
	done
	@echo "=== SASS dump complete ==="

# ------------------------------------------------------------------
# Reproducibility entry points.
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

# Full-corpus "run everything" pass (#124). Builds the whole corpus, then
# runs every bench and records every result + metadata to
# results/bench_all/<timestamp>/ (results.json, summary.md, samples.jsonl).
# On-demand data collection -- NOT the regression gate (that stays `make
# bench` / bench_regress.R, unchanged). Skip nothing, record everything.
# Extra flags: make bench-all ARGS="--min-valid 3 --max-attempts 10"
# Plan only (no GPU): make bench-all ARGS=--list   (or run bench_all.R directly)
#
# The build is best-effort (`-k` keep-going, `-` ignore failure): a corpus
# bench that can't compile (e.g. a reference bench on a box without
# cuSPARSELt/cuDNN) must NOT abort the pass -- bench_all.R records it
# `not-built` and continues. Aborting here would defeat the whole
# skip-nothing principle (the recipe would never run).
bench-all:
	-@$(MAKE) -k all
	@$(RSCRIPT) scripts/bench/bench_all.R $(ARGS)

figures:
	@echo "=== Regenerating docs/figures ==="
	@$(RSCRIPT) scripts/audit/generate_readme_figures.R
	@$(RSCRIPT) scripts/profile/roofline_measured.R
	@$(RSCRIPT) scripts/audit/sass_histogram.R
	@$(RSCRIPT) scripts/cymatic/cymatic_visualize.R

bench-reference: reference
	@$(RSCRIPT) scripts/bench/bench_reference.R

compare-reference:
	@$(RSCRIPT) scripts/bench/compare_reference.R

reference-pipeline: reference bench-reference compare-reference

# Publish the kernel corpus to the Hugging Face dataset repo
# pjt222/ga104-cuda-kernels. Rebuilds the corpus first (GPU required)
# and needs HF_TOKEN in .env. Dry-run: make publish-hf ARGS=--dry-run
publish-hf:
	@$(RSCRIPT) scripts/publish_hf.R $(ARGS)

reproduce: setup verify all bench figures
	@echo ""
	@echo "================================================================"
	@echo "Full reproduction complete."
	@echo "  setup    -- R deps installed via renv"
	@echo "  verify   -- toolchain + GPU detected"
	@echo "  all      -- every cubin + bench compiled"
	@echo "  bench    -- results compared to data/baselines.json"
	@echo "  figures  -- docs/figures regenerated"
	@echo "================================================================"

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
clean:
	@echo "Cleaning generated files (git-tracked artifacts preserved)..."
	@find . -path ./tools -prune -o -type f \
	  \( -name '*.$(SM_ARCH).cubin' -o -name '*.cuasm' -o -name '*.sass' \
	     -o -name '*.reassembled.cubin' \) -print | \
	  while IFS= read -r f; do \
	    git ls-files --error-unmatch "$$f" >/dev/null 2>&1 || rm -f "$$f"; \
	  done
	@for exe in $(BENCH_EXES); do \
	  git ls-files --error-unmatch "$$exe" >/dev/null 2>&1 \
	    || rm -f "$$exe" 2>/dev/null || true; \
	done
	@git ls-files --error-unmatch kernels/tutorial/host >/dev/null 2>&1 \
	  || rm -f kernels/tutorial/host 2>/dev/null || true
	@echo "Done."

# ------------------------------------------------------------------
# Help
# ------------------------------------------------------------------
help:
	@echo "bare-metal GPU build system"
	@echo ""
	@echo "Reproducibility:"
	@echo "  make reproduce — one-stop: setup + verify + build + bench + figures"
	@echo "  make setup     — renv::restore() + install cuasmR"
	@echo "  make verify    — environment check (CUDA, GPU, cuasmR)"
	@echo "  make bench     — run benches vs data/baselines.json"
	@echo "  make figures   — regenerate docs/figures via R scripts"
	@echo "  make bench-reference — run local reference benches vs data/reference_baselines.json"
	@echo "  make compare-reference — compare project baselines to local reference baselines"
	@echo "  make reference-pipeline — build + validate + compare local reference benches"
	@echo ""
	@echo "Publish:"
	@echo "  make publish-hf — rebuild + upload kernel corpus to the HF dataset repo"
	@echo "  make publish-hf ARGS=--dry-run — resolve the upload manifest only (no build/upload)"
	@echo ""
	@echo "Build:"
	@echo "  make all       — build all cubins + benches"
	@echo "  make cubins    — compile all .cu files to .cubin"
	@echo "  make benches   — compile all bench*.cu to executables"
	@echo "  make test      — run smoke tests on compiled benches"
	@echo "  make disasm    — disassemble all cubins to .sass"
	@echo "  make clean     — remove all generated artifacts"
	@echo ""
	@echo "By family:"
	@echo "  make tutorial      — vector_add hello-world"
	@echo "  make gemm          — sgemm/hgemm/hgemm_sparse/igemm"
	@echo "  make reductions    — softmax/layernorm/groupnorm"
	@echo "  make elementwise   — activations/timestep_emb"
	@echo "  make attention     — flash_attention/cross_attention"
	@echo "  make convolution   — conv2d/resblock"
	@echo "  make memory_layout — cymatic"
	@echo "  make composition   — attention_layer"
	@echo "  make reference     — local reference-library benches"
	@echo ""
	@echo "  make help      — show this message"

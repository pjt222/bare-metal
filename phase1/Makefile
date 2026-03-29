# Phase 1 Makefile — vector_add SASS hello world
#
# Usage:
#   make cubin        — compile kernel to .cubin
#   make disasm       — disassemble to .cuasm + .sass
#   make host         — compile the host driver
#   make roundtrip    — verify CuAssembler roundtrip is stable
#   make all          — cubin + disasm + host
#   make clean        — remove generated files

SM_ARCH  := sm_86
NVCC     := nvcc
NVCC_FLAGS := --cubin -arch=$(SM_ARCH) -O1
PYTHON   := python

KERNEL_SRC  := vector_add.cu
CUBIN       := vector_add.$(SM_ARCH).cubin
CUASM       := vector_add.$(SM_ARCH).cuasm
SASS        := vector_add.$(SM_ARCH).sass
HOST_SRC    := host.cu
HOST_EXE    := host.exe

.PHONY: all cubin disasm host roundtrip clean

all: cubin disasm host

cubin: $(CUBIN)

$(CUBIN): $(KERNEL_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "Cubin: $@"

disasm: $(CUBIN)
	$(PYTHON) ../scripts/build.py disasm $(CUBIN)

host: $(HOST_EXE)

$(HOST_EXE): $(HOST_SRC)
	$(NVCC) -arch=$(SM_ARCH) -o $@ $< -lcuda
	@echo "Host: $@"

roundtrip: $(CUBIN)
	$(PYTHON) ../scripts/build.py roundtrip $(KERNEL_SRC)

clean:
	del /Q $(CUBIN) $(CUASM) $(SASS) $(HOST_EXE) *.reassembled.cubin *.modified.cubin 2>nul || true

NVCC ?= nvcc
TARGET := cuda_cpp_interview_lab

CUDA_ARCH ?= 89
NVCC_ARCH_FLAGS := \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)
NVCC_FLAGS := -O3 -std=c++17 -lineinfo --use_fast_math -Xcompiler -Wall,-Wextra $(NVCC_ARCH_FLAGS)
INCLUDES := -Iinclude
SOURCES := src/main.cu src/pipeline_cpu.cpp src/pipeline_gpu.cu

.PHONY: all run profile clean help

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	mkdir -p reports
	ncu --set default \
		--kernel-name-base demangled \
		--kernel-name regex:.*(add_bias_relu|row_sum|normalize|row_argmax).* \
		-o reports/pipeline_default \
		./$(TARGET)

clean:
	rm -f $(TARGET)

help:
	@echo "Targets:"
	@echo "  make          - Build"
	@echo "  make run      - Build and run"
	@echo "  make profile  - Run Nsight Compute profile"
	@echo "  make clean    - Remove binary"
	@echo "Variables:"
	@echo "  CUDA_ARCH=89  - Compile for compute/sm 89 (default)"
	@echo "Examples:"
	@echo "  make CUDA_ARCH=89"
	@echo "  make CUDA_ARCH=120   # if your nvcc supports it"

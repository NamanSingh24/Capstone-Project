# Makefile — GPU Capstone: CUDA Image Filter Pipeline
#
# Adjust GPU_ARCH for your card:
#   sm_60=Pascal  sm_75=Turing  sm_80=Ampere  sm_86=RTX30xx  sm_89=Ada  sm_90=Hopper
GPU_ARCH   ?= sm_75
STREAMS    ?= 4
INPUT_DIR  ?= data
OUTPUT_DIR ?= output
LOG_DIR    ?= logs

NVCC      := nvcc
CUDA_PATH ?= $(shell dirname $(shell which nvcc) 2>/dev/null)/..

NVCC_FLAGS := -arch=$(GPU_ARCH) -std=c++14 -O3 --use_fast_math \
              -I src -I $(CUDA_PATH)/include

LD_FLAGS   := -L $(CUDA_PATH)/lib64 -lcudart

BUILD_DIR  := build
TARGET     := capstone

CUDA_OBJS  := $(BUILD_DIR)/kernels.cu.o \
              $(BUILD_DIR)/processor.cu.o
CXX_OBJS   := $(BUILD_DIR)/image_io.cpp.o \
              $(BUILD_DIR)/main.cpp.o

.PHONY: all clean run_demo stb_headers

all: stb_headers $(TARGET)

$(TARGET): $(CUDA_OBJS) $(CXX_OBJS)
	$(NVCC) -arch=$(GPU_ARCH) -o $@ $^ $(LD_FLAGS)

$(BUILD_DIR)/%.cu.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

$(BUILD_DIR)/%.cpp.o: src/%.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -x c++ -o $@ -c $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Download stb single-file headers into src/ if not present
stb_headers: src/stb_image.h src/stb_image_write.h

src/stb_image.h:
	curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -o $@

src/stb_image_write.h:
	curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" -o $@

run_demo: all
	mkdir -p $(OUTPUT_DIR) $(LOG_DIR)
	./$(TARGET) --input $(INPUT_DIR) --output $(OUTPUT_DIR) \
	            --logs $(LOG_DIR) --filter all --streams $(STREAMS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

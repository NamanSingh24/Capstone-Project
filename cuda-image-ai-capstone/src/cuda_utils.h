// cuda_utils.h — CUDA error-checking macros and GPU timer.
// Part of: GPU Capstone — CUDA Real-Time Video Frame Processor
#ifndef CAPSTONE_SRC_CUDA_UTILS_H_
#define CAPSTONE_SRC_CUDA_UTILS_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking macro: aborts on any CUDA API failure.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                        \
              __FILE__, __LINE__, cudaGetErrorString(_err));                 \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// ---------------------------------------------------------------------------
// Scoped CUDA event timer.  Measures wall-clock GPU time in milliseconds.
// ---------------------------------------------------------------------------
struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start(cudaStream_t s = 0) { cudaEventRecord(start, s); }
  float Stop(cudaStream_t s = 0) {
    cudaEventRecord(stop, s);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};

#endif  // CAPSTONE_SRC_CUDA_UTILS_H_

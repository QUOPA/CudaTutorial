#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>

#define CHECK_CUDA_ERROR(err) \
do { if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

class ScopedEventRecorder {
public:
  ScopedEventRecorder() {
    CHECK_CUDA_ERROR(cudaEventCreate(&_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(_start, 0));
  }

  ~ScopedEventRecorder() {
    CHECK_CUDA_ERROR(cudaEventRecord(_stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(_stop));
    float elapsed_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, _start, _stop));
    printf("Kernel run time: %3.3f ms\n", elapsed_time);
  }

private:
  cudaEvent_t _start;
  cudaEvent_t _stop;

};

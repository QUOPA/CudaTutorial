#include <cuda_runtime.h>
#include <cuda.h>
#include "testcommon/testmacros.h"
#include "CudaCommonUtil.cuh"
#include <memory>


namespace constmemory {

__constant__ int b_constmem[1];

__global__ void _CUDAIncrementKernel(int* d_a, size_t arr_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < arr_size) {
    d_a[idx] += *b_constmem;
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void _CUDAIncrementNoConstKernel(int* d_a, int *d_b, size_t arr_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < arr_size) {
    d_a[idx] += *d_b;
    idx += blockDim.x * gridDim.x;
  }
}

void CUDAIncrement(int* h_a, size_t arr_size) {
  ScopedEventRecorder scoped_timer;
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)& d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int h_b = 10;
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(b_constmem, &h_b, sizeof(int)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAIncrementKernel << < 256, 256 >> > , d_a, arr_size);

  CHECK_CUDA_ERROR(cudaMemcpy(h_a, d_a, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));

}

void CUDAIncrementNoConst(int* h_a, size_t arr_size) {
  ScopedEventRecorder scoped_timer;
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int h_b = 10;
  int* d_b = nullptr;
  
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAIncrementNoConstKernel << < 256, 256 >> > , d_a, d_b, arr_size);

  CHECK_CUDA_ERROR(cudaMemcpy(h_a, d_a, arr_size * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
}

}
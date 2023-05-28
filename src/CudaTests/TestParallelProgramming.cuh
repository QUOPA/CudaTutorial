#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "testcommon/testmacros.h"
#include "CudaCommonUtil.cuh"

namespace parallelprogramming {

__global__ void _CUDAParalleladd_Kernel(int* d_a, int* d_b, int* d_c, size_t arr_size) {
  int idx = blockIdx.x;
  if (idx < arr_size) {
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

__global__ void _CUDAParalleladd_Kernel_Threads(int* d_a, int* d_b, int* d_c, size_t arr_size) {
  int idx = threadIdx.x;
  if (idx < arr_size) {
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

__global__ void _CUDAParalleladd_Kernel_Blocks_and_Threads(int* d_a, int* d_b, int* d_c, size_t arr_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < arr_size) {
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

__global__ void _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids(int* d_a, int* d_b, int* d_c, size_t arr_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < arr_size) {
    d_c[idx] = d_a[idx] + d_b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

void CUDAParalleladd(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(int)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAParalleladd_Kernel << <arr_size, 1 >> >, d_a, d_b, d_c, arr_size);

  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
//  for (int idx = 0; idx < arr_size; idx++) {
//    std::cout << h_c[idx] << ", ";
//  }
//  std::cout << std::endl;
}

void CUDAParalleladd_Threads(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(int)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAParalleladd_Kernel_Threads << <1, arr_size >> > , d_a, d_b, d_c, arr_size);

  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  //  for (int idx = 0; idx < arr_size; idx++) {
  //    std::cout << h_c[idx] << ", ";
  //  }
  //  std::cout << std::endl;
}

void CUDAParalleladd_Blocks_and_Threads(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(int)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAParalleladd_Kernel_Blocks_and_Threads << < (arr_size + 511)/512, 512 >> > , d_a, d_b, d_c, arr_size);


  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  //  for (int idx = 0; idx < arr_size; idx++) {
  //    std::cout << h_c[idx] << ", ";
  //  }
  //  std::cout << std::endl;
}

void CUDAParalleladd_Blocks_and_Threads_Grids(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(int)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAParalleladd_Kernel_Blocks_and_Threads_Grids << < 16, 16>> > , d_a, d_b, d_c, arr_size);


  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  //  for (int idx = 0; idx < arr_size; idx++) {
  //    std::cout << h_c[idx] << ", ";
  //  }
  //  std::cout << std::endl;
}




} //namespace parallelprogramming
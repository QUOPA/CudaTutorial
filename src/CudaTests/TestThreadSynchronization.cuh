#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "testcommon/testmacros.h"
#include "CudaCommonUtil.cuh"
#include <memory>
static constexpr int threadsPerBlock = 512;

namespace threadsynchronization {

__global__ void _dotprodcache(float *d_a, float *d_b, float *d_c, size_t arr_size) {
  __shared__ float cache[threadsPerBlock];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (idx < arr_size) {
    temp += d_a[idx] * d_b[idx];
    idx += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = temp;
  __syncthreads();

  int i = threadsPerBlock / 2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  
  if (cacheIndex == 0)
    d_c[blockIdx.x] = cache[cacheIndex];

}

__global__ void _vecMultiplication(float* d_a, float* d_b, float* d_c, size_t arr_size) {
  __shared__ float cache[threadsPerBlock];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  while (idx < arr_size) {
    d_c[idx] += d_a[idx] * d_b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

void DotProductCache(float* h_a, float* h_b, size_t arr_size) {
  float* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(float), cudaMemcpyHostToDevice));

  float* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(float), cudaMemcpyHostToDevice));

  int required_blockes = (arr_size + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks = 16 < required_blockes ? 16 : required_blockes;

  float* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, numBlocks * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemset(d_c, 0.0,numBlocks * sizeof(float)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_dotprodcache << < numBlocks, threadsPerBlock >> > , d_a, d_b, d_c, arr_size);

  auto h_c = std::make_unique<float[]>(numBlocks);
  CHECK_CUDA_ERROR(cudaMemcpy(h_c.get(), d_c, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

  float the_value = 0;
  for (int idx = 0; idx < numBlocks; idx++) {
    the_value += h_c[idx];
  }

  float expected = arr_size -1;
  expected = expected * (expected + 1) * (2 * expected + 1) / 6 * 100;
  std::cout << "Calculated value is: " << the_value << std::endl;
  std::cout << "Expected valie is: " << expected << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
}

void DotProduct(float* h_a, float* h_b, size_t arr_size) {
  float* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(float), cudaMemcpyHostToDevice));

  float* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(float), cudaMemcpyHostToDevice));

  int required_blockes = (arr_size + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks = 16 < required_blockes ? 16 : required_blockes;

  float* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(float)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_vecMultiplication << < numBlocks, threadsPerBlock >> > , d_a, d_b, d_c, arr_size);

  auto h_c = std::make_unique<float[]>(arr_size);
  CHECK_CUDA_ERROR(cudaMemcpy(h_c.get(), d_c, arr_size * sizeof(float), cudaMemcpyDeviceToHost));

  float the_value = 0;
  for (int idx = 0; idx < arr_size; idx++) {
    the_value += h_c[idx];
  }

  float expected = arr_size - 1;
  expected = expected * (expected + 1) * (2 * expected + 1) / 6 * 100;
  std::cout << "Calculated value is: " << the_value << std::endl;
  std::cout << "Expected valie is: " << expected << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
}




}
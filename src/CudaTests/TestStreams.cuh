#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "testcommon/testmacros.h"
#include "CudaCommonUtil.cuh"
#include <memory>

namespace streams {
 
__global__ void _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids(int* d_a, int* d_b, int* d_c, size_t arr_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < arr_size) {
    d_c[idx] = d_a[idx] + d_b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

void AsyncOneStream(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice, stream));

  int* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, h_b, arr_size * sizeof(int), cudaMemcpyHostToDevice, stream));

  int* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(int)));

  _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids <<< 256, 256, 0, stream >>> (d_a, d_b, d_c, arr_size);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c, d_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
  
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));


  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
}

void AsyncTwoStreams(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  cudaStream_t stream0;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream0));
  cudaStream_t stream1;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));


  int* d_a0 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a0, arr_size / 2 * sizeof(int)));
  int* d_a1 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a1, arr_size / 2 * sizeof(int)));
  int* d_b0 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b0, arr_size / 2 * sizeof(int)));
  int* d_b1 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b1, arr_size / 2 * sizeof(int)));
  int* d_c0 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c0, arr_size / 2 * sizeof(int)));
  int* d_c1 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c1, arr_size / 2 * sizeof(int)));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a0, h_a, arr_size/2 * sizeof(int), cudaMemcpyHostToDevice, stream0));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b0, h_b, arr_size/2 * sizeof(int), cudaMemcpyHostToDevice, stream0));

  _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids << < 256, 256, 0, stream0 >> > (d_a0, d_b0, d_c0, arr_size/2);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c, d_c0, arr_size / 2 * sizeof(int), cudaMemcpyDeviceToHost, stream0));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a1, h_a + arr_size / 2, arr_size / 2 * sizeof(int), cudaMemcpyHostToDevice, stream1));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b1, h_b + arr_size / 2, arr_size / 2 * sizeof(int), cudaMemcpyHostToDevice, stream1));

  _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids << < 256, 256, 0, stream1 >> > (d_a1, d_b1, d_c1, arr_size/2);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c + arr_size / 2, d_c1, arr_size / 2 * sizeof(int), cudaMemcpyDeviceToHost, stream1));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream0));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream0));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));


  CHECK_CUDA_ERROR(cudaFree(d_a0));
  CHECK_CUDA_ERROR(cudaFree(d_b0));
  CHECK_CUDA_ERROR(cudaFree(d_c0));

  CHECK_CUDA_ERROR(cudaFree(d_a1));
  CHECK_CUDA_ERROR(cudaFree(d_b1));
  CHECK_CUDA_ERROR(cudaFree(d_c1));
}

void AsyncTwoStreamsOptimized(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  cudaStream_t stream0;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream0));
  cudaStream_t stream1;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));


  int* d_a0 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a0, arr_size / 2 * sizeof(int)));
  int* d_a1 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a1, arr_size / 2 * sizeof(int)));
  int* d_b0 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b0, arr_size / 2 * sizeof(int)));
  int* d_b1 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b1, arr_size / 2 * sizeof(int)));
  int* d_c0 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c0, arr_size / 2 * sizeof(int)));
  int* d_c1 = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c1, arr_size / 2 * sizeof(int)));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a0, h_a, arr_size / 2 * sizeof(int), cudaMemcpyHostToDevice, stream0));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a1, h_a + arr_size / 2, arr_size / 2 * sizeof(int), cudaMemcpyHostToDevice, stream1));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b0, h_b, arr_size / 2 * sizeof(int), cudaMemcpyHostToDevice, stream0));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b1, h_b + arr_size / 2, arr_size / 2 * sizeof(int), cudaMemcpyHostToDevice, stream1));

  _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids << < 256, 256, 0, stream0 >> > (d_a0, d_b0, d_c0, arr_size / 2);
  _CUDAParalleladd_Kernel_Blocks_and_Threads_Grids << < 256, 256, 0, stream1 >> > (d_a1, d_b1, d_c1, arr_size / 2);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c, d_c0, arr_size / 2 * sizeof(int), cudaMemcpyDeviceToHost, stream0));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c + arr_size / 2, d_c1, arr_size / 2 * sizeof(int), cudaMemcpyDeviceToHost, stream1));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream0));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream0));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));


  CHECK_CUDA_ERROR(cudaFree(d_a0));
  CHECK_CUDA_ERROR(cudaFree(d_b0));
  CHECK_CUDA_ERROR(cudaFree(d_c0));

  CHECK_CUDA_ERROR(cudaFree(d_a1));
  CHECK_CUDA_ERROR(cudaFree(d_b1));
  CHECK_CUDA_ERROR(cudaFree(d_c1));
}


}
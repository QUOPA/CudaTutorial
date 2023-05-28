#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "testcommon/testmacros.h"
#include "CudaCommonUtil.cuh"

namespace cudahelloworld
{

__global__ void hellokernel() {
}

__global__ void _CUDAadd_Kernel(int* d_a, int* d_b, int* d_c) {
  for (int i = 0; i < 50000; i++) {
    d_c[i] = d_a[i] + d_b[i];
  }
}

void CUDAadd(int* h_a, int* h_b, int* h_c, size_t arr_size) {
  int* d_a = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_b = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, arr_size * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, arr_size * sizeof(int), cudaMemcpyHostToDevice));

  int* d_c = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, arr_size * sizeof(int)));

  REGISTERTESTSUBCASE_CUDA_KERNEL(_CUDAadd_Kernel << <1, 1 >> > ,d_a, d_b, d_c);

  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));

  //for (int idx = 0; idx < arr_size; idx++) {
  //  std::cout << h_c[idx] << ", ";
  //}
  //std::cout << std::endl;
}

void TestCudaGetDeviceCount() {
  int count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&count));
  std::cout << "Available devices: " << count << std::endl;
}

void TestShowAllDeviceInfo() {
  cudaDeviceProp prop;
  int count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
    printf(" --- General Information for device %d ---\n", i);
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Device copy overlap: ");
    if (prop.deviceOverlap)
      printf("Enabled\n");
    else
      printf("Disabled\n");
    printf("Kernel execition timeout : ");
    if (prop.kernelExecTimeoutEnabled)
      printf("Enabled\n");
    else
      printf("Disabled\n");
    printf(" --- Memory Information for device %d ---\n", i);
    printf("Total global mem: %ld\n", prop.totalGlobalMem);
    printf("Total constant Mem: %ld\n", prop.totalConstMem);
    printf("Max mem pitch: %ld\n", prop.memPitch);
    printf("Texture Alignment: %ld\n", prop.textureAlignment);
    printf(" --- MP Information for device %d ---\n", i);
    printf("Multiprocessor count: %d\n",
      prop.multiProcessorCount);
    printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
    printf("Registers per mp: %d\n", prop.regsPerBlock);
    printf("Threads in warp: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n",
      prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n",
      prop.maxThreadsDim[0], prop.maxThreadsDim[1],
      prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n",
      prop.maxGridSize[0], prop.maxGridSize[1],
      prop.maxGridSize[2]);
    printf("\n");
  }
}

} //namespace cudahelloworld
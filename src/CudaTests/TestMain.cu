#include "testcommon/testmacros.h"
#include "TestSuites.cuh"
#include <iostream>
#include <memory>

int main()
{
  constexpr size_t arr_size = 50000;
  auto a = std::make_unique<int[]>(arr_size);
  auto b = std::make_unique<int[]>(arr_size);
  auto c = std::make_unique<int[]>(arr_size);

  for (int idx = 0; idx < arr_size; idx++) {
    a[idx] = idx;
    b[idx] = idx * 100;
  }
  
  REGISTERTESTSUITE(cudahelloworld, a.get(), b.get(), c.get(), arr_size);
  REGISTERTESTSUITE(parallelprogramming, a.get(), b.get(), c.get(), arr_size);


  constexpr size_t arr_size_dotprod = 50000;
  auto fa = std::make_unique<float[]>(arr_size_dotprod);
  auto fb = std::make_unique<float[]>(arr_size_dotprod);
 
  for (int idx = 0; idx < arr_size_dotprod; idx++) {
    fa[idx] = idx;
    fb[idx] = idx * 100;
  }

  REGISTERTESTSUITE(threadsynchronization, fa.get(), fb.get(), arr_size_dotprod);

  REGISTERTESTSUITE_NOARG(constmemory);

  int* ha = nullptr;
  int* hb = nullptr;
  int* hc = nullptr;

  constexpr size_t arr_size_streams = 5E8;
  CHECK_CUDA_ERROR(cudaHostAlloc((void**)&ha, arr_size_streams * sizeof(int), cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc((void**)&hb, arr_size_streams * sizeof(int), cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc((void**)&hc, arr_size_streams * sizeof(int), cudaHostAllocDefault));
  for (int idx = 0; idx < arr_size_streams; idx++) {
    ha[idx] = idx;
    hb[idx] = idx * 100;
  }

  REGISTERTESTSUITE(streams, ha, hb, hc, arr_size_streams);

  CHECK_CUDA_ERROR(cudaFreeHost(ha));
  CHECK_CUDA_ERROR(cudaFreeHost(hb));
  CHECK_CUDA_ERROR(cudaFreeHost(hc));

  char buff[256];
  std::cin.getline(buff, 256);
  return 0;
}
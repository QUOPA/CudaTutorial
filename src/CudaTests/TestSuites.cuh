#pragma once

#include "cuda_runtime.h"
#include "TestHelloWorld.cuh"
#include "TestParallelProgramming.cuh"
#include "TestThreadSynchronization.cuh"
#include "TestConstMemory.cuh"
#include "TestStreams.cuh"

namespace cudahelloworld {
  void TestSuite(int* h_a, int* h_b, int* h_c, size_t arr_size) {
    REGISTERTESTCASE_CUDA_KERNEL(hellokernel << <1, 1 >> > );
    REGISTERTESTCASE(CUDAadd, h_a, h_b, h_c, arr_size);
    REGISTERTESTCASE(TestCudaGetDeviceCount);
    REGISTERTESTCASE(TestShowAllDeviceInfo);
  }
}

namespace parallelprogramming {
  void TestSuite(int* h_a, int* h_b, int* h_c, size_t arr_size) {
    REGISTERTESTCASE(CUDAParalleladd, h_a, h_b, h_c, arr_size);
    REGISTERTESTCASE(CUDAParalleladd_Threads, h_a, h_b, h_c, arr_size);
    REGISTERTESTCASE(CUDAParalleladd_Blocks_and_Threads, h_a, h_b, h_c, arr_size);
    REGISTERTESTCASE(CUDAParalleladd_Blocks_and_Threads_Grids, h_a, h_b, h_c, arr_size);
  }
}

namespace threadsynchronization {
  void TestSuite(float* h_a, float* h_b, size_t arr_size) {
    REGISTERTESTCASE(DotProductCache, h_a, h_b, arr_size);
    REGISTERTESTCASE(DotProduct, h_a, h_b, arr_size);

  }
}

namespace constmemory {
  void TestSuite() {
    constexpr size_t arr_size_increment_test = 50000;
    auto a_incr_test = std::make_unique<int[]>(arr_size_increment_test);
    for (int idx = 0; idx < arr_size_increment_test; idx++) {
      a_incr_test[idx] = idx;
    }
    REGISTERTESTCASE(CUDAIncrement, a_incr_test.get(), arr_size_increment_test);
    for (int idx = 0; idx < arr_size_increment_test; idx++) {
      a_incr_test[idx] = idx;
    }
    REGISTERTESTCASE(CUDAIncrementNoConst, a_incr_test.get(), arr_size_increment_test);
  }
}

namespace streams {
  void TestSuite(int* h_a, int* h_b, int* h_c, size_t arr_size) {
    REGISTERTESTCASE(AsyncOneStream, h_a, h_b, h_c, arr_size);
    REGISTERTESTCASE(AsyncTwoStreams, h_a, h_b, h_c, arr_size);
    REGISTERTESTCASE(AsyncTwoStreamsOptimized, h_a, h_b, h_c, arr_size);
  }
}
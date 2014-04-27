#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define cudaSafeCall(a) {                                 \
    a;                                                    \
    cudaError e = cudaGetLastError();                     \
    if (e != cudaSuccess) {                               \
      printf("Cuda Error: %s\n", cudaGetErrorString(e));  \
    }                                                     \
}


#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cula.h>
#include <stdio.h>

#define MAX_CHARS 1000

#define cudaSafeCall(a) {                                 \
    a;                                                    \
    cudaError e = cudaGetLastError();                     \
    if (e != cudaSuccess) {                               \
      printf("Cuda Error: %s\n", cudaGetErrorString(e));  \
      exit(0);						  \
    }                                                     \
}

#define cublasSafeCall(a) {                                 \
    cublasStatus_t status = a;                                                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      printf("Cublas Error: %d\n", status);  \
      exit(0);						  \
    }                                                     \
}

#define culaSafeCall(a) {                                 \
    culaStatus status = a;                                                    \
    if (status != culaNoError) {					\
      char buffer[MAX_CHARS];						\
      culaGetErrorInfoString(status, culaGetErrorInfo(), buffer, MAX_CHARS); \
      printf("Cula Error: %s\n", buffer);				\
      exit(0);						  \
    }                                                     \
}


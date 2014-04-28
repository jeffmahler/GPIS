#include "gpu_active_set_selector.hpp"

#include "cuda_macros.h"

#include "active_set_buffers.h"
#include "classification_buffers.h"
#include "max_subset_buffers.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

bool GpuActiveSetSelector::Select(int maxSize, float* inputPoints, float* targetPoints,
				  GpuActiveSetSelector::SubsetSelectionMode mode,
				  GaussianProcessHyperparams hypers,
				  int inputDim, int targetDim, int numPoints, float tolerance,
				  float* activeInputs, float* activeTargets)
{
  // allocate matrices / vectors for computations
  float* d_activeKernelMatrix; // active set kernel matrix
  float* d_activeInputs;    // active set target vector
  float* d_activeTargets;   // active set target vector
  float* d_kernelVector;    // kernel vector
  float* d_p; // conjugate gradient conjugate vector
  float* d_q; // conjugate gradient auxiliary vector
  float* d_r; // conjugate gradient residual vector
  cudaSafeCall(cudaMalloc((void**)d_activeKernelMatrix, maxSize * maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_activeInputs, inputDim * maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_activeTargets, targetDim * maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_kernelVector, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_p, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_q, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_r, maxSize * sizeof(float)));

  // allocate auxiliary buffers
  ActiveSetBuffers activeSetBuffers;
  MaxSubsetBuffers maxSubBuffers;
  ClassificationBuffers classificationBuffers;
  construct_active_set_buffers(&activeSetBuffers, inputDim, targetDim, maxSize);
  construct_max_subset_buffers(&maxSubBuffers, inputPoints, targetPoints, inputDim, targetDim, numPoints);
  construct_classification_buffers(&classificationBuffers, numPoints);
  
  // init random starting point and update the buffers
  int firstIndex = rand() % numPoints;
  activate_max_subset_buffers(&maxSubBuffers, firstIndex);
  update_active_set_buffers_cpu(&activeSetBuffers, inputPoints, targetPoints, firstIndex, hypers);
 
  // beta is the scaling of the variance when classifying points
  float beta = 2 * log(numPoints * pow(M_PI,2) / (6 * tolerance));  

  for (unsigned int k = 1; k < maxSize; k++) {
    // compute alpha vector

    // compute kernel vectors


    // compute variance helper vectors

    // compute amibugity

    // max ambiguity reduction (and update of active set)

    // update above, below classification

    // update matrices

    // update beta
    beta = 2 * log(numPoints * pow(M_PI,2) * pow((k+1),2) / (6 * tolerance));
  }

  // free everything
  cudaSafeCall(cudaFree(d_activeKernelMatrix));
  cudaSafeCall(cudaFree(d_activeInputs));
  cudaSafeCall(cudaFree(d_activeTargets));
  cudaSafeCall(cudaFree(d_kernelVector));
  cudaSafeCall(cudaFree(d_p));
  cudaSafeCall(cudaFree(d_q));
  cudaSafeCall(cudaFree(d_r));

  free_active_set_buffers(&activeSetBuffers);
  free_max_subset_buffers(&maxSubBuffers);
  free_classification_buffers(&classificationBuffers);

  return true;
}

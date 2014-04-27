#include "active_subset_selector.hpp"

#include "cuda_macros.h"
#include "classification_buffers.h"
#include "max_subset_buffers.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

#define POINT_INDEX(x, dim) ((dim)*x)

bool ActiveSubsetSelector::Select(int maxSize, float* inputPoints, float* targetPoints,
				  ActiveSubsetSelector::SubsetSelectionMode mode,
				  GaussianProcessHyperparams hypers,
				  int inputDim, int targetDim, int numPoints, float tolerance,
				  float* activeInputs, float* activeTargets)
{
  float* A; // active set kernel matrix
  float* r; // conjugate gradient residual vector
  float* p; // conjugate gradient conjugate vector
  float* q; // conjugate gradient auxiliary vector
  float* alpha;

  // allocate auxiliary buffers
  int numActive = 0;
  float* d_ambiguity;
  float* d_activeInputs;
  float* d_activeTargets;
  MaxSubsetBuffers maxSubBuffers;
  ClassificationBuffers classificationBuffers;
  cudaSafeCall(cudaMalloc((void**)d_ambiguity, numPoints * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_activeInputs, inputDim * numPoints * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_activeTargets, targetDim * numPoints * sizeof(float)));
  construct_max_subset_buffers(&maxSubBuffers, inputPoints, targetPoints, inputDim, targetDim, numPoints);
  construct_classification_buffers(&classificationBuffers, numPoints);
  
  // init random starting point
  int firstIndex = rand() % numPoints;
  int pointIndex = POINT_INDEX(firstIndex, inputDim);
  cudaSafeCall(cudaMemset((void**)(maxSubBuffers.active + firstIndex), 1, sizeof(unsigned char)));

  // copy result to gpu input and target buffers
  for (int i = 0; i < inputDim; i++) {
    activeInputs[numActive+i] = inputPoints[pointIndex+i]; 
  }
  for (int i = 0; i < targetDim; i++) {
    activeTargets[numActive+i] = targetPoints[pointIndex+i]; 
  }
  numActive++;

  // allocate the kernel vector


  // beta is the scaling of the variance when classifying points
  float beta = 2 * log(numPoints * pow(M_PI,2) / (6 * tolerance));  

  for (unsigned int k = 1; k < maxSize; k++) {
    // compute kernel vectors

    // compute alpha vector

    // compute variance helper vectors

    // compute amibugity

    // max ambiguity reduction (and update of active set)

    // update above, below classification

    // update matrices

    // update beta
    beta = 2 * log(numPoints * pow(M_PI,2) * pow((k+1),2) / (6 * tolerance));
  }

  cudaSafeCall(cudaFree(d_ambiguity));
  cudaSafeCall(cudaFree(d_activeInputs));
  cudaSafeCall(cudaFree(d_activeTargets));
  free_max_subset_buffers(&maxSubBuffers);
  free_classification_buffers(&classificationBuffers);

  return true;
}

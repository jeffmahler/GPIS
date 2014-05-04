// Class to encapsulate selection of the active subset

#pragma once

#include <cublas_v2.h>
#include <string>
#include <vector>

#include "active_set_buffers.h"

class GpuActiveSetSelector {

  // possible criteria from which to select the active subset
  enum SubsetSelectionMode {
    ENTROPY,
    LEVEL_SET
  };

 public:
  GpuActiveSetSelector() {}
  ~GpuActiveSetSelector() {}

 public:
  bool SelectFromGrid(const std::string& csvFilename, int setSize, float sigma, float beta,
		      int width, int height, int depth, bool storeDepth = true);
  // Select an active subset from 
  bool Select(int maxSize, float* inputPoints, float* targetPoints,
	      SubsetSelectionMode mode,
	      GaussianProcessHyperparams hypers,
	      int inputDim, int targetDim, int numPoints, float tolerance,
	      float* activeInputs, float* activeTargets);

 private:
  bool ReadCsv(const std::string& csvFilename, int width, int height, int depth, bool storeDepth,
	       float* inputs, float* targets);
  float SECovariance(float* x, float* y, int dim, int sigma);

 private:
  bool GpPredict(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers, int index, GaussianProcessHyperparams hypers, float* d_kernelVector, float* d_p, float* d_q, float* d_r, float* d_alpha, float* d_gamma, float tolerance, cublasHandle_t* handle, float* d_mu, float* d_sigma);
  bool SolveLinearSystem(ActiveSetBuffers* activeSetBuffers, float* target, float* d_p, float* d_q, float* d_r, float* d_alpha, float tolerance, cublasHandle_t* handle);
};

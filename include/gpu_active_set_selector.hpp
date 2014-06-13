// Class to encapsulate selection of the active subset

#pragma once

#include <cublas_v2.h>
#include <string>
#include <vector>

#include "active_set_buffers.h"

struct PredictionError {
  float mean;
  float std;
  float median;
  float min;
  float max;
};

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
		      int width, int height, int depth, int batchSize, float tolerance,
		      bool storeDepth = false);
  // Select an active subset from 
  bool SelectCG(int maxSize, float* inputPoints, float* targetPoints,
		SubsetSelectionMode mode,
		GaussianProcessHyperparams hypers,
		int inputDim, int targetDim, int numPoints, float tolerance,
		float* activeInputs, float* activeTargets);

  // Select an active subset from 
  bool SelectChol(int maxSize, float* inputPoints, float* targetPoints,
		  SubsetSelectionMode mode,
		  GaussianProcessHyperparams hypers,
		  int inputDim, int targetDim, int numPoints, float tolerance, int batchSize,
		  float* activeInputs, float* activeTargets);

 private:
  float SECovariance(float* x, float* y, int dim, int sigma);
  double ReadTimer();
  bool WriteCsv(const std::string& csvFilename, float* buffer, int width, int height);
  bool ReadCsv(const std::string& csvFilename, int width, int height, int depth, bool storeDepth,
	       float* inputs, float* targets);
  bool EvaluateErrors(float* d_mu, float* d_targets, unsigned char* d_active, int numPts,
		      PredictionError& errorStruct);

 private:
  bool GpPredictCG(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers, int index,
		   GaussianProcessHyperparams hypers, float* d_kernelVector, float* d_p, float* d_q,
		   float* d_r, float* d_alpha, float* d_gamma, float* d_scalar1, float* d_scalar2, 
		   float tolerance, cublasHandle_t* handle, float* d_mu, float* d_sigma);
  bool SolveLinearSystemCG(ActiveSetBuffers* activeSetBuffers, float* target, float* d_p, float* d_q,
			 float* d_r, float* d_alpha, float* d_scalar1, float* d_scalar2,
			 float tolerance, cublasHandle_t* handle);

 private:
  bool GpPredictChol(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers,
		     int index, GaussianProcessHyperparams hypers, float* d_kernelVector,
		     float* d_L, float* d_alpha, float* d_gamma,
		     cublasHandle_t* handle, float* d_mu, float* d_sigma);
  bool GpPredictCholBatch(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers,
			  int index, int batchSize, GaussianProcessHyperparams hypers,
			  float* d_kernelVectors, float* d_L, float* d_alpha, float* d_gamma,
			  float* d_scalar1, float* d_scalar2, cublasHandle_t* handle, float* d_mu, float* d_sigma);
  bool SolveLinearSystemChol(ActiveSetBuffers* activeSetBuffers, float* target, float* d_L,
			     float* d_alpha, cublasHandle_t* handle);
  

 private:
  double checkpoint_;
  double elapsed_;
};

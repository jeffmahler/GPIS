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

/**
 * @class GpuActiveSetSelector
 * @brief Selects an 'active' set of points to be used in constructing a Gaussian Process Implicit Surface
 */
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
  /**
   * SelectFromGrid
   * @brief Selects an 'active' set of points to be used to construct a GPIS for the TSDF in the specified CSV
   */
  bool SelectFromGrid(const std::string& csvFilename, int setSize, float sigma, float beta,
		      int width, int height, int depth, int batchSize, float tolerance,
		      float accuracy, bool storeDepth = false, int startIndex = -1);
  /**
   * SelectCG
   * @brief Selects the active set using CG to solve the linear system
   */
  bool SelectCG(int maxSize, float* inputPoints, float* targetPoints,
		SubsetSelectionMode mode,
		GaussianProcessHyperparams hypers,
		int inputDim, int targetDim, int numPoints, float tolerance,
		float* activeInputs, float* activeTargets);

  /**
   * SelectChol
   * @brief Selects the active set using a cholesky decomposition to solve the linear system
   */
  bool SelectChol(int maxSize, float* inputPoints, float* targetPoints,
		  SubsetSelectionMode mode,
		  GaussianProcessHyperparams hypers,
		  int inputDim, int targetDim, int numPoints, float tolerance, float accuracy, int batchSize,
		  float* activeInputs, float* activeTargets, int startIndex = -1, bool incremental = true);

  bool SelectFullInversion(float* inputs, float* targets, int inputDim, int targetDim, int numPoints, 
                           GaussianProcessHyperparams hypers);


 private:
  float SECovariance(float* x, float* y, int dim, int sigma);
  double ReadTimer();
  double Duration();
  bool WriteCsv(const std::string& csvFilename, float* buffer, int width, int height, int lda);
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
			     float* d_alpha, float* d_scalar1, cublasHandle_t* handle);
  bool UpdateChol(ActiveSetBuffers* activeSetBuffers, float* target, float* d_L,
                  float* d_alpha, float* d_gamma, float* d_x, float* d_scalar1, 
                  GaussianProcessHyperparams hypers, cublasHandle_t* handle);

 private:
  double checkpoint_;
  double elapsed_;
};

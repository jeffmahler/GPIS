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
  // initialize cublas
  cublasHandle_t handle;
  cublasCreate(&handle);

  // allocate matrices / vectors for computations
  float* d_kernelVector;    // kernel vector
  float* d_p; // conjugate gradient conjugate vector
  float* d_q; // conjugate gradient auxiliary vector
  float* d_r; // conjugate gradient residual vector
  float* d_alpha; // vector representing the solution to the mean equation of GPR
  float* d_gamma; // auxiliary vector to receive the kernel vector product
  float* d_mu;
  float* d_sigma;

  cudaSafeCall(cudaMalloc((void**)d_kernelVector, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_p, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_q, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_r, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_alpha, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_gamma, maxSize * sizeof(float)));

  cudaSafeCall(cudaMalloc((void**)d_mu, numPoints * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)d_sigma, numPoints * sizeof(float)));

  // allocate auxiliary buffers
  ActiveSetBuffers activeSetBuffers;
  MaxSubsetBuffers maxSubBuffers;
  ClassificationBuffers classificationBuffers;
  construct_active_set_buffers(&activeSetBuffers, inputDim, targetDim, maxSize);
  construct_max_subset_buffers(&maxSubBuffers, inputPoints, targetPoints, inputDim, targetDim, numPoints);
  construct_classification_buffers(&classificationBuffers, numPoints);
  
  // allocate host buffers for determining set to check
  unsigned char* h_active = new unsigned char[numPoints];
  unsigned char* h_upper = new unsigned char[numPoints];
  unsigned char* h_lower = new unsigned char[numPoints];

  // init random starting point and update the buffers
  int firstIndex = rand() % numPoints;
  activate_max_subset_buffers(&maxSubBuffers, firstIndex);
  update_active_set_buffers(&activeSetBuffers, &maxSubBuffers, firstIndex, hypers);
 
  // beta is the scaling of the variance when classifying points
  float beta = 2 * log(numPoints * pow(M_PI,2) / (6 * tolerance));  
  float level = 0;
  int nextIndex = 0;

  for (unsigned int k = 1; k < maxSize; k++) {
    // compute alpha vector
    SolveLinearSystem(&activeSetBuffers, activeSetBuffers.active_targets, d_p, d_q, d_r, d_alpha, tolerance, &handle); 

    // get the current classification statuses from the GPU
    cudaSafeCall(cudaMemcpy(h_active, maxSubBuffers.active, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_upper, classificationBuffers.upper, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_lower, classificationBuffers.lower, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // predict all active points
    for (unsigned int i = 0; i < numPoints; i++) {
      if (!h_active[i] && !h_upper[i] && !h_lower[i]) {
	GpPredict(&maxSubBuffers, &activeSetBuffers, i, hypers,
		  d_kernelVector, d_p, d_q, d_r, d_alpha, d_gamma,
		  tolerance, &handle, d_mu, d_sigma); 
      }
    }

    // compute amibugity and max ambiguity reduction (and update of active set)
    nextIndex = find_best_active_set_candidate(&maxSubBuffers, &classificationBuffers,
					       d_mu, d_sigma, level, beta, hypers);

    // update matrices
    update_active_set_buffers(&activeSetBuffers, &maxSubBuffers, nextIndex, hypers);

    // update beta according to formula in level set probing paper
    beta = 2 * log(numPoints * pow(M_PI,2) * pow((k+1),2) / (6 * tolerance));
  }

  // free everything
  cudaSafeCall(cudaFree(d_kernelVector));
  cudaSafeCall(cudaFree(d_p));
  cudaSafeCall(cudaFree(d_q));
  cudaSafeCall(cudaFree(d_r));
  cudaSafeCall(cudaFree(d_alpha));
  cudaSafeCall(cudaFree(d_gamma));

  cudaSafeCall(cudaFree(d_mu));
  cudaSafeCall(cudaFree(d_sigma));

  free_active_set_buffers(&activeSetBuffers);
  free_max_subset_buffers(&maxSubBuffers);
  free_classification_buffers(&classificationBuffers);

  delete [] h_active;
  delete [] h_upper;
  delete [] h_lower;

  cublasDestroy(handle);

  return true;
}

float GpuActiveSetSelector::SECovariance(float* x, float* y, int dim, int sigma)
{
  float sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return exp(-sum / (2 * sigma));

}

bool GpuActiveSetSelector::GpPredict(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers, int index, GaussianProcessHyperparams hypers, float* kernelVector, float* d_p, float* d_q, float* d_r, float* d_alpha, float* d_gamma, float tolerance, cublasHandle_t* handle, float* d_mu, float* d_sigma)
{
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;
  int numPts = subsetBuffers->num_pts;

  compute_kernel_vector(activeSetBuffers, subsetBuffers, index, kernelVector, hypers);
  SolveLinearSystem(activeSetBuffers, kernelVector, d_p, d_q, d_r, d_gamma, tolerance, handle); 

  // store the predicitve mean in mu
  cublasSdot(*handle, maxActive, d_alpha, 1, kernelVector, 1, d_mu + index);  

  // store the variance REDUCTION in sigma, not the actual variance
  cublasSdot(*handle, maxActive, d_gamma, 1, kernelVector, 1, d_sigma + index);  
}

bool GpuActiveSetSelector::SolveLinearSystem(ActiveSetBuffers* activeSetBuffers, float* target, float* d_p, float* d_q, float* d_r, float* d_alpha, float tolerance, cublasHandle_t* handle)
{
  // iterative conjugate gradient
  int k = 0;
  float scale = 1.0f;
  float s = 0.0f;
  float t = 0.0f;
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;
  float delta_0, delta_1;

  cudaSafeCall(cudaMemset(d_alpha, 0, maxActive * sizeof(float)));
  cudaSafeCall(cudaMemcpy(d_r, target, maxActive * sizeof(float), cudaMemcpyDeviceToDevice)); 
  cudaSafeCall(cudaMemcpy(d_p, d_r, maxActive * sizeof(float), cudaMemcpyDeviceToDevice)); 

  // get intial residual
  cublasSdot(*handle, maxActive, d_r, 1, d_r, 1, &delta_0);
  delta_1 = delta_0;

  // solve for the next conjugate vector until tolerance is satisfied
  while (delta_1 > tolerance && k < maxActive) {
    // q = Ap
    cublasSgemv(*handle, CUBLAS_OP_N, maxActive, maxActive, &scale, activeSetBuffers->active_kernel_matrix, maxActive, d_p, 1, 0, d_q, 1);

    // s = p^T q 
    cublasSdot(*handle, maxActive, d_p, 1, d_q, 1, &s);

    t = delta_1 / s;

    // alpha = alpha + t * p
    cublasSaxpy(*handle, maxActive, &t, d_p, 1, d_alpha, 1); 

    // r = r - t * q
    t = -1 * t;
    cublasSaxpy(*handle, maxActive, &t, d_q, 1, d_r, 1); 

    // delta_1 = r^T r
    delta_0 = delta_1;
    cublasSdot(*handle, maxActive, d_r, 1, d_r, 1, &delta_1);

    // p = r + beta * p
    s = delta_0 / delta_1;
    cublasSaxpy(*handle, maxActive, &s, d_r, 1, d_p, 1);
    s = 1.0f / s;
    cublasSscal(*handle, maxActive, &s, d_p, 1);
  }

  return true;
}

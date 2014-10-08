/*
 * Class implementation for GPU-accelerated active set selection for GPIS
 *
 * @author Jeff Mahler
 * @email  jmahler@berkeley.edu
*/
#include "gpu_active_set_selector.hpp"

#include "cuda_macros.h"

#include "active_set_buffers.h"
#include "classification_buffers.h"
#include "max_subset_buffers.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cula.h>
#include <cula_lapack.h>
#include <cula_lapack_device.h>

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <sys/time.h>
#include <time.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/moment.hpp>

float GpuActiveSetSelector::SECovariance(float* x, float* y, int dim, int sigma)
{
  float sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return exp(-sum / (2 * sigma));
}

double GpuActiveSetSelector::ReadTimer()
{
  static bool initialized = false;
  static struct timeval start;
  struct timeval end;
  if( !initialized )
    {
      gettimeofday( &start, NULL );
      initialized = true;
    }
  gettimeofday( &end, NULL );
  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

double GpuActiveSetSelector::Duration()
{
  double checkpoint = elapsed_;
  elapsed_ = ReadTimer();
  return (elapsed_ - checkpoint);
}

bool GpuActiveSetSelector::WriteCsv(const std::string& csvFilename, float* buffer, int width, int height, int lda)
{
  std::ofstream csvFile(csvFilename.c_str());
  std::string delim = ",";
  float* hostBuffer = new float[width * lda];
  cudaSafeCall(cudaMemcpy(hostBuffer, buffer, width * lda * sizeof(float), cudaMemcpyDeviceToHost));


  // if (width == 2) {
  //   for (int i = 0; i < width*height; i++) {
  //     std::cout << hostBuffer[i] << std::endl;
  //   }
  // }

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      csvFile << hostBuffer[j + i*lda];
      if (i < width-1)
	csvFile << delim;
    }
    csvFile << "\n";
  }
  csvFile.close();

  delete [] hostBuffer;
  return true;
}

bool GpuActiveSetSelector::ReadCsv(const std::string& csvFilename, int width, int height, int depth, bool storeDepth,
	     float* inputs, float* targets)
{
  std::ifstream csvFile(csvFilename.c_str());
  if (!csvFile.is_open()) {
    LOG(ERROR) << "Failed to open " << csvFilename;
    return false;
  }

 int maxChars = 10000;
  char buffer[maxChars];

  int j = 0;
  int k = 0;
  float val = 0.0f;
  int numPts = width*height*depth;
  char delim;

  while(!csvFile.eof() && k < depth) {
    csvFile.getline(buffer, maxChars);

    std::stringstream parser(buffer);
    for (int i = 0; i < width; i++) {
      parser >> val;
      if (i < width-1)
	parser >> delim;
      inputs[IJK_TO_LINEAR(i, j, k, width, height) + 0 * numPts] = i;
      inputs[IJK_TO_LINEAR(i, j, k, width, height) + 1 * numPts] = j;
      if (storeDepth) {
	inputs[IJK_TO_LINEAR(i, j, k, width, height) + 2 * numPts] = k;
      }
      targets[IJK_TO_LINEAR(i, j, k, width, height) + 0 * numPts] = val;
      //      std::cout << i << " " << j << " " << k << " " << val << std::endl;
    }

    // set the next index
    j++;
    if (j >= height) {
      j = 0;
      k++;
    }
  }
  
  csvFile.close();
  return true;
}

bool GpuActiveSetSelector::EvaluateErrors(float* d_mu, float* d_targets, unsigned char* d_active, int numPts, PredictionError& errorStruct)

{					       
  unsigned char* active = new unsigned char[numPts];
  float* predictions = new float[numPts];
  float* targets = new float[numPts];

  cudaSafeCall(cudaMemcpy(active, d_active, numPts * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  cudaSafeCall(cudaMemcpy(predictions, d_mu, numPts * sizeof(float), cudaMemcpyDeviceToHost));
  cudaSafeCall(cudaMemcpy(targets, d_targets, numPts * sizeof(float), cudaMemcpyDeviceToHost));

  boost::accumulators::accumulator_set<float, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::moment<2> > > meanAccumulator;
  boost::accumulators::accumulator_set<float, boost::accumulators::stats<boost::accumulators::tag::median > > medianAccumulator;
  boost::accumulators::accumulator_set<float, boost::accumulators::stats<boost::accumulators::tag::min, boost::accumulators::tag::max > > maxMinAccumulator;

  int numTest = 0;
  float absError = 0.0f;

  for (int i = 0; i < numPts; i++) {
    // std::cout << "Point " << i << std::endl;
    // std::cout << "Pred " << predictions[i] << std::endl;
    // std::cout << "Target " << targets[i] << std::endl;
    if (active[i] == 0) {
      absError = fabs(predictions[i] - targets[i]);
      // std::cout << "Point " << i << std::endl;
      // std::cout << "Prediction:" << predictions[i] << std::endl;
      // std::cout << "Actual:" << targets[i] << std::endl;
      // std::cout << "Error:" << absError << std::endl;
      meanAccumulator(absError);
      medianAccumulator(absError);
      maxMinAccumulator(absError);
      numTest++;
    }
    else {
      //      std::cout << "Target " << targets[i] <<  " pred " << predictions[i] << std::endl;
    }
  }

  errorStruct.mean = 0;
  errorStruct.std = 0;
  errorStruct.median = 0;
  errorStruct.max = 0;
  errorStruct.min = 0;
  if (numTest > 0) {
    errorStruct.mean = boost::accumulators::mean(meanAccumulator);
    errorStruct.std = boost::accumulators::moment<2>(meanAccumulator);
    errorStruct.median = boost::accumulators::median(medianAccumulator);
    errorStruct.max = boost::accumulators::max(maxMinAccumulator);
    errorStruct.min = boost::accumulators::min(maxMinAccumulator);
  }

  delete [] active;
  delete [] predictions;
  delete [] targets;
  return true;
}

bool GpuActiveSetSelector::SelectFromGrid(const std::string& csvFilename, int setSize, float sigma, float beta,
					  int width, int height, int depth, int batchSize, 
					  float tolerance, bool storeDepth)
{
  // read in csv
  int inputDim = 2;
  int targetDim = 1;
  if (storeDepth) {
    inputDim = 3;
  }

  int numPts = width*height*depth;
  float* inputs = new float[numPts * inputDim];
  float* targets = new float[numPts * targetDim];
  float* activeInputs = new float[numPts * inputDim];
  float* activeTargets = new float[numPts * targetDim];
  GaussianProcessHyperparams hypers;
  hypers.beta = beta;
  hypers.sigma = sigma;

  VLOG(1) << "Reading TSDF from " << csvFilename;
  if(!ReadCsv(csvFilename, width, height, depth, storeDepth, inputs, targets)) {
    return false;
  }

  SelectChol(setSize, inputs, targets, GpuActiveSetSelector::LEVEL_SET, hypers, inputDim, targetDim, numPts, tolerance, batchSize, activeInputs, activeTargets);
  //SelectCG(setSize, inputs, targets, GpuActiveSetSelector::LEVEL_SET, hypers, inputDim, targetDim, numPts, tolerance, activeInputs, activeTargets);

  delete [] inputs;
  delete [] targets;
  delete [] activeInputs;
  delete [] activeTargets;  

  return true;
}

bool GpuActiveSetSelector::SelectCG(int maxSize, float* inputPoints, float* targetPoints,
				    GpuActiveSetSelector::SubsetSelectionMode mode,
				    GaussianProcessHyperparams hypers,
				    int inputDim, int targetDim, int numPoints, float tolerance,
				    float* activeInputs, float* activeTargets)
{
  // initialize cula
  culaSafeCall(culaInitialize());

  // initialize cublas
  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));
  cublasSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

  // allocate matrices / vectors for computations
  float* d_kernelVector;    // kernel vector
  float* d_p; // conjugate gradient conjugate vector
  float* d_q; // conjugate gradient auxiliary vector
  float* d_r; // conjugate gradient residual vector
  float* d_alpha; // vector representing the solution to the mean equation of GPR
  float* d_gamma; // auxiliary vector to receive the kernel vector product
  float* d_mu;
  float* d_sigma;
  float* d_scalar1;
  float* d_scalar2;

  // force valid num points
  if (maxSize > numPoints) {
    maxSize = numPoints;
  }

  VLOG(1) << "Allocating device memory...";
  cudaSafeCall(cudaMalloc((void**)&d_kernelVector, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_p, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_q, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_r, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_alpha, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_gamma, maxSize * sizeof(float)));

  cudaSafeCall(cudaMalloc((void**)&d_mu, numPoints * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_sigma, numPoints * sizeof(float)));

  cudaSafeCall(cudaMalloc((void**)&d_scalar1, sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_scalar2, sizeof(float)));

  // allocate auxiliary buffers
  VLOG(1) << "Allocating device buffers...";
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
  VLOG(1) << "Setting first index...";
  int firstIndex = rand() % numPoints;
  VLOG(1) << "Chose " << firstIndex << " as first index ";
  activate_max_subset_buffers(&maxSubBuffers, firstIndex);
  update_active_set_buffers(&activeSetBuffers, &maxSubBuffers, hypers);

  elapsed_ = 0.0f;
 
  // compute initial alpha vector
  SolveLinearSystemCG(&activeSetBuffers, activeSetBuffers.active_targets, d_p, d_q, d_r, d_alpha, 
		      d_scalar1, d_scalar2, tolerance, &handle); 
  checkpoint_ = Duration();
  VLOG(1) << "CG Time (sec):\t " << checkpoint_;

  // beta is the scaling of the variance when classifying points
  float beta = 2 * log(numPoints * pow(M_PI,2) / (6 * tolerance));  
  float level = 0;
  int numLeft = numPoints - 1;
  
  VLOG(1) << "Using beta  = " << beta;

  for (unsigned int k = 1; k < maxSize && numLeft > 0; k++) {
    VLOG(1) << "Selecting point " << k+1 << "...";
    
    // get the current classification statuses from the GPU
    cudaSafeCall(cudaMemcpy(h_active, maxSubBuffers.active, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_upper, classificationBuffers.upper, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_lower, classificationBuffers.lower, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    checkpoint_ = Duration();
    VLOG(1) << "Memcpy Time (sec):\t " << checkpoint_;

    // predict all active points
    numLeft = 0;
    for (unsigned int i = 0; i < numPoints; i++) {
      if (!h_active[i] && !h_upper[i] && !h_lower[i]) {
	numLeft++;
	GpPredictCG(&maxSubBuffers, &activeSetBuffers, i, hypers,
		    d_kernelVector, d_p, d_q, d_r, d_alpha, d_gamma,
		    d_scalar1, d_scalar2,
		    tolerance, &handle, d_mu, d_sigma); 
      }
    }
    if (numLeft == 0) {
      continue;
    }
    VLOG(1) << "Num left " << numLeft;

    checkpoint_ = Duration();
    VLOG(1) << "Prediction Time (sec):\t " << checkpoint_;

    // compute amibugity and max ambiguity reduction (and update of active set)
    find_best_active_set_candidate(&maxSubBuffers, &classificationBuffers,
				   d_mu, d_sigma, level, beta, hypers);

    checkpoint_ = Duration();
    VLOG(1) << "Reduction Time (sec):\t " << checkpoint_;

    // update matrices
    //    activate_max_subset_buffers(&maxSubBuffers, nextIndex);
    update_active_set_buffers(&activeSetBuffers, &maxSubBuffers, hypers);

    checkpoint_ = Duration();
    VLOG(1) << "Update Time (sec):\t " << checkpoint_;

    // update beta according to formula in level set probing paper
    beta = 2 * log(numPoints * pow(M_PI,2) * pow((k+1),2) / (6 * tolerance));

    // compute next alpha vector
    SolveLinearSystemCG(&activeSetBuffers, activeSetBuffers.active_targets, d_p, d_q, d_r, d_alpha,
			d_scalar1, d_scalar2, tolerance, &handle); 

    checkpoint_ = Duration();
    VLOG(1) << "CG Solve Time (sec):\t " << checkpoint_;
  }

  VLOG(1) << std::endl;
  VLOG(1) << "Done selecting active set";
  VLOG(1) << "Set Selection Took " << elapsed_ << " sec. ";

  // predict all points and compute the error
  VLOG(1) << "Computing errors...";
  for (unsigned int i = 0; i < numPoints; i++) {
    //    VLOG(1) << "Predicting " << i << "...";
    GpPredictCG(&maxSubBuffers, &activeSetBuffers, i, hypers,
		d_kernelVector, d_p, d_q, d_r, d_alpha, d_gamma,
		d_scalar1, d_scalar2, 
		tolerance, &handle, d_mu, d_sigma); 
  }
  VLOG(1) << "All predicted...";
  PredictionError errors;
  EvaluateErrors(d_mu, maxSubBuffers.targets, maxSubBuffers.active, numPoints, errors);
  VLOG(1) << "Error statistics";
  VLOG(1) << "Mean:\t" << errors.mean;
  VLOG(1) << "Std:\t" << errors.std;
  VLOG(1) << "Median:\t" << errors.median;
  VLOG(1) << "Min:\t" << errors.min;
  VLOG(1) << "Max:\t" << errors.max;

  // save everything
  WriteCsv("inputs.csv", activeSetBuffers.active_inputs, activeSetBuffers.dim_input, activeSetBuffers.num_active, maxSize);
  WriteCsv("targets.csv", activeSetBuffers.active_targets, activeSetBuffers.dim_target, activeSetBuffers.num_active, maxSize);
  WriteCsv("alpha.csv", d_alpha, 1, activeSetBuffers.num_active, maxSize);

  // free everything
  cudaSafeCall(cudaFree(d_kernelVector));
  cudaSafeCall(cudaFree(d_p));
  cudaSafeCall(cudaFree(d_q));
  cudaSafeCall(cudaFree(d_r));
  cudaSafeCall(cudaFree(d_alpha));
  cudaSafeCall(cudaFree(d_gamma));

  cudaSafeCall(cudaFree(d_mu));
  cudaSafeCall(cudaFree(d_sigma));

  cudaSafeCall(cudaFree(d_scalar1));
  cudaSafeCall(cudaFree(d_scalar2));

  free_active_set_buffers(&activeSetBuffers);
  free_max_subset_buffers(&maxSubBuffers);
  free_classification_buffers(&classificationBuffers);

  delete [] h_active;
  delete [] h_upper;
  delete [] h_lower;

  cublasDestroy(handle);
  culaShutdown();

  return true;
}

bool GpuActiveSetSelector::SelectChol(int maxSize, float* inputPoints, float* targetPoints,
				      GpuActiveSetSelector::SubsetSelectionMode mode,
				      GaussianProcessHyperparams hypers,
				      int inputDim, int targetDim, int numPoints, float tolerance,
				      int batchSize, float* activeInputs, float* activeTargets, bool incremental)
{
  VLOG(1) << "Selcting active set using Cholesky decomposition";

  // initialize cula
  culaSafeCall(culaInitialize());

  VLOG(1) << "Initialized cula";

  // initialize cublas
  cublasHandle_t handle;
  cublasSafeCall(cublasCreate(&handle));
  cublasSafeCall(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

  VLOG(1) << "Initialized cublas";

  // allocate matrices / vectors for computations
  float* d_kernelVector;    // kernel vector
  float* d_L; // Cholesky factor
  float* d_alpha; // vector representing the solution to the mean equation of GPR
  float* d_gamma; // auxiliary vector to receive the kernel vector product
  float* d_scalar1;
  float* d_scalar2;
  float* d_mu;
  float* d_sigma;

  // force valid num points
  if (maxSize > numPoints) {
    maxSize = numPoints;
  }

  VLOG(1) << "Using max size " << maxSize;
  VLOG(1) << "Allocating device memory...";
  cudaSafeCall(cudaMalloc((void**)&d_kernelVector, maxSize * batchSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_L, maxSize * maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_alpha, maxSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_gamma, maxSize * batchSize * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_scalar1, sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_scalar2, sizeof(float)));

  cudaSafeCall(cudaMalloc((void**)&d_mu, numPoints * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&d_sigma, numPoints * sizeof(float)));

  float scale = 1.0f;
  float zero = 0.0f;
  cudaSafeCall(cudaMemcpy(d_scalar1, &scale, sizeof(float), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpy(d_scalar2, &zero, sizeof(float), cudaMemcpyHostToDevice));

  // allocate auxiliary buffers
  VLOG(1) << "Allocating device buffers...";
  ActiveSetBuffers activeSetBuffers;
  MaxSubsetBuffers maxSubBuffers;
  ClassificationBuffers classificationBuffers;
  construct_active_set_buffers(&activeSetBuffers, inputDim, targetDim, maxSize);
  construct_max_subset_buffers(&maxSubBuffers, inputPoints, targetPoints, inputDim, targetDim, numPoints);
  construct_classification_buffers(&classificationBuffers, numPoints);
  
  // allocate host buffers for determining set to check
  // unsigned char* h_active = new unsigned char[numPoints];
  // unsigned char* h_upper = new unsigned char[numPoints];
  // unsigned char* h_lower = new unsigned char[numPoints];

  // init random starting point and update the buffers
  VLOG(1) << "Setting first index...";
  int firstIndex = rand() % numPoints;
  VLOG(1) << "Chose " << firstIndex << " as first index ";
  activate_max_subset_buffers(&maxSubBuffers, firstIndex);
  update_active_set_buffers(&activeSetBuffers, &maxSubBuffers, hypers);

  elapsed_ = 0.0f;
 
  // compute initial alpha vector
  VLOG(1) << "Solving initial linear system";

  //  if (!incremental) 
  SolveLinearSystemChol(&activeSetBuffers, activeSetBuffers.active_targets, d_L, d_alpha,
			&handle); 
  // float hostL;
  // cudaSafeCall(cudaMemcpy(&hostL, d_L, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "Received L " << hostL;


  checkpoint_ = Duration();
  VLOG(1) << "Chol Time (sec):\t " << checkpoint_;

  // beta is the scaling of the variance when classifying points
  float beta = 2 * log(numPoints * pow(M_PI,2) / (6 * tolerance));  
  float level = 0.0f;
  float score = 0.0f;
  int nextIndex = 0;
  std::vector<int> indices;
  indices.push_back(firstIndex);

  VLOG(1) << "Using beta  = " << beta;

  unsigned int k;
  for (k = 1; k < maxSize && nextIndex != -1; k++) {
    VLOG(1) << "Selecting point " << k+1 << " of " << maxSize << "...";
    
    // get the current classification statuses from the GPU
    // cudaSafeCall(cudaMemcpy(h_active, maxSubBuffers.active, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // cudaSafeCall(cudaMemcpy(h_upper, classificationBuffers.upper, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // cudaSafeCall(cudaMemcpy(h_lower, classificationBuffers.lower, numPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    // int numLeft = 0;
    // for (unsigned int i = 0; i < numPoints; i++) {
    //   if (!h_active[i] && !h_upper[i] && !h_lower[i]) {
    //     numLeft++;
    //   }
    // }
    // VLOG(1) << "Points not classified: " << numLeft;

    // checkpoint_ = Duration();
    // VLOG(1) << "Memcpy Time (sec):\t " << checkpoint_;

    // predict all active points
    for (int i = 0; i < numPoints; i += batchSize) {
      GpPredictCholBatch(&maxSubBuffers, &activeSetBuffers, i, std::min(batchSize, numPoints - i),
			 hypers, d_kernelVector, d_L, d_alpha, d_gamma, d_scalar1, d_scalar2,
			 &handle, d_mu, d_sigma); 
    }

    checkpoint_ = Duration();
    VLOG(1) << "Prediction Time (sec):\t " << checkpoint_;

    // compute amibugity and max ambiguity reduction (and update of active set)
    find_best_active_set_candidate(&maxSubBuffers, &classificationBuffers,
				   d_mu, d_sigma, level, beta, hypers);
    cudaSafeCall(cudaMemcpy(&nextIndex, maxSubBuffers.d_next_index, sizeof(int), cudaMemcpyDeviceToHost)); 
    cudaSafeCall(cudaMemcpy(&score, maxSubBuffers.scores, sizeof(float), cudaMemcpyDeviceToHost)); 

    // only update the active set if points were left
    indices.push_back(nextIndex);

    VLOG(1) << "Chose " << nextIndex << " with score " << score << " as next index...";
    if (nextIndex >= 0) {

      checkpoint_ = Duration();
      VLOG(1) << "Reduction Time (sec):\t " << checkpoint_;

      // update matrices
      //    activate_max_subset_buffers(&maxSubBuffers, nextIndex);
      update_active_set_buffers(&activeSetBuffers, &maxSubBuffers, hypers);

      //    WriteCsv("M.csv", activeSetBuffers.active_kernel_matrix, activeSetBuffers.num_active, maxSize);

      checkpoint_ = Duration();
      VLOG(1) << "Update Time (sec):\t " << checkpoint_;

      // update beta according to formula in level set probing paper
      beta = 2 * log(numPoints * pow(M_PI,2) * pow((k+1),2) / (6 * tolerance));

      // compute next alpha vector
      SolveLinearSystemChol(&activeSetBuffers, activeSetBuffers.active_targets, d_L, d_alpha,
                            &handle); 

      checkpoint_ = Duration();
      VLOG(1) << "Chol Solve Time (sec):\t " << checkpoint_;

      VLOG(1) << std::endl;
    }
  }

  VLOG(1) << std::endl;
  VLOG(1) << "Done selecting active set";
  LOG(INFO) << "Selection of " <<  k << " points took " << elapsed_ << " sec. ";

  // predict all points and compute the error
  VLOG(1) << "Computing errors...";
  for (int i = 0; i < numPoints; i += batchSize) {
    GpPredictCholBatch(&maxSubBuffers, &activeSetBuffers, i, std::min(batchSize, numPoints - i),
		       hypers, d_kernelVector, d_L, d_alpha, d_gamma, d_scalar1, d_scalar2,
		       &handle, d_mu, d_sigma); 
  }
  VLOG(1) << "All predicted...";
  PredictionError errors;
  EvaluateErrors(d_mu, maxSubBuffers.targets, maxSubBuffers.active, numPoints, errors);
  VLOG(1) << "Error statistics";
  VLOG(1) << "Mean:\t" << errors.mean;
  VLOG(1) << "Std:\t" << errors.std;
  VLOG(1) << "Median:\t" << errors.median;
  VLOG(1) << "Min:\t" << errors.min;
  VLOG(1) << "Max:\t" << errors.max;

       // for(int i = 0; i < indices.size(); i++) {
       //   std::cout << indices[i] << std::endl;
       // }

  // save everything
       WriteCsv("inputs.csv", activeSetBuffers.active_inputs, activeSetBuffers.dim_input, activeSetBuffers.num_active, maxSize);
       WriteCsv("targets.csv", activeSetBuffers.active_targets, activeSetBuffers.dim_target, activeSetBuffers.num_active, maxSize);
       WriteCsv("alpha.csv", d_alpha, 1, activeSetBuffers.num_active, maxSize);
       WriteCsv("predictions.csv", d_mu, 1, numPoints, numPoints);

  // free everything
  cudaSafeCall(cudaFree(d_kernelVector));
  cudaSafeCall(cudaFree(d_L));
  cudaSafeCall(cudaFree(d_alpha));
  cudaSafeCall(cudaFree(d_gamma));
  cudaSafeCall(cudaFree(d_scalar1));
  cudaSafeCall(cudaFree(d_scalar2));

  cudaSafeCall(cudaFree(d_mu));
  cudaSafeCall(cudaFree(d_sigma));

  free_active_set_buffers(&activeSetBuffers);
  free_max_subset_buffers(&maxSubBuffers);
  free_classification_buffers(&classificationBuffers);

  // delete [] h_active;
  // delete [] h_upper;
  // delete [] h_lower;

  cublasDestroy(handle);
  culaShutdown();

  return true;
}

bool GpuActiveSetSelector::GpPredictCG(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers, int index,
				       GaussianProcessHyperparams hypers, float* kernelVector, float* d_p, float* d_q,
				       float* d_r, float* d_alpha, float* d_gamma, float* d_scalar1, float* d_scalar2, 
				       float tolerance, cublasHandle_t* handle, float* d_mu, float* d_sigma)
{
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;
  int numPts = subsetBuffers->num_pts;

  compute_kernel_vector(activeSetBuffers, subsetBuffers, index, kernelVector, hypers);

  checkpoint_ = elapsed_;
  elapsed_ = ReadTimer();
  checkpoint_ = elapsed_ - checkpoint_;
  VLOG(1) << "KV Time (sec):\t " << checkpoint_;

  SolveLinearSystemCG(activeSetBuffers, kernelVector, d_p, d_q, d_r, d_gamma, d_scalar1, d_scalar2, tolerance, handle); 

  checkpoint_ = Duration();
  VLOG(1) << "SLS Time (sec):\t " << checkpoint_;

  // store the predicitve mean in mu
  cublasSafeCall(cublasSdot(*handle, numActive, d_alpha, 1, kernelVector, 1, &(d_mu[index])));  

  // float hostK[2];
  // cudaSafeCall(cudaMemcpy(hostK, kernelVector, 2*sizeof(float), cudaMemcpyDeviceToHost));
  // float hostAlpha[2];
  // cudaSafeCall(cudaMemcpy(hostAlpha, d_alpha, 2*sizeof(float), cudaMemcpyDeviceToHost));
  // float hostMu;
  // cudaSafeCall(cudaMemcpy(&hostMu, &(d_mu[index]), sizeof(float), cudaMemcpyDeviceToHost));
  // float hostT[2];
  // float hostM[4];
  // cudaSafeCall(cudaMemcpy(&hostT, activeSetBuffers->active_targets, 2*sizeof(float), cudaMemcpyDeviceToHost));
  // cudaSafeCall(cudaMemcpy(&hostM, activeSetBuffers->active_kernel_matrix, 4*sizeof(float), cudaMemcpyDeviceToHost));

  // std::cout << "KV " << hostK[0] << " " << hostK[1] << std::endl;
  // std::cout << "KM " << hostM[0] << " " << hostM[1] << " " << hostM[2] << " " << hostM[3] << std::endl;
  // std::cout << "Targets " << hostT[0] << " " << hostT[1] << std::endl;
  // std::cout << "Alpha " << hostAlpha[0] << " " << hostAlpha[1] << std::endl;
  // std::cout << "mu " << hostMu << std::endl;

  // store the variance REDUCTION in sigma, not the actual variance
  cublasSafeCall(cublasSdot(*handle, numActive, d_gamma, 1, kernelVector, 1, &(d_sigma[index])));

  float hostSig;
  cudaSafeCall(cudaMemcpy(&hostSig, &(d_sigma[index]), sizeof(float), cudaMemcpyDeviceToHost));
  VLOG(1) << "Sig at " << index << " : " << hostSig;

  checkpoint_ = Duration();
  VLOG(1) << "Dot Time (sec):\t " << checkpoint_;
}

bool GpuActiveSetSelector::SolveLinearSystemCG(ActiveSetBuffers* activeSetBuffers, float* target, float* d_p, float* d_q, float* d_r, float* d_alpha, float* d_scalar1, float* d_scalar2, float tolerance, cublasHandle_t* handle)
{
  // iterative conjugate gradient
  int k = 0;
  float s = 0.0f;
  float t = 0.0f;
  float delta_0 = 0.0f;
  float delta_1 = 0.0f;
  float scale_1 = 1.0f;
  float scale_2 = 0.0f;
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;

  //  std::cout << std::endl << "CG SOLVE" << std::endl;

  // set initial values
  cudaSafeCall(cudaMemset(d_alpha, 0, maxActive * sizeof(float)));
  cudaSafeCall(cudaMemcpy(d_r, target, maxActive * sizeof(float), cudaMemcpyDeviceToDevice)); 
  cudaSafeCall(cudaMemcpy(d_p, d_r, maxActive * sizeof(float), cudaMemcpyDeviceToDevice)); 
  // float hostR;
  // cudaSafeCall(cudaMemcpy(&hostR, d_r, sizeof(float), cudaMemcpyDeviceToHost));
  //  std::cout << "R " << hostR << " " << maxActive << std::endl;

  // checkpoint_ = elapsed_;
  // elapsed_ = ReadTimer();
  // checkpoint_ = elapsed_ - checkpoint_;
  // std::cout << "CuSet Time (sec):\t " << checkpoint_ << std::endl;

  // get intial residual
  cublasSafeCall(cublasSdot(*handle, maxActive, d_r, 1, d_r, 1, &(d_scalar1[0])));
  cudaSafeCall(cudaMemcpy(&delta_0, d_scalar1, sizeof(float), cudaMemcpyDeviceToHost));
  delta_1 = delta_0;

  // checkpoint_ = elapsed_;
  // elapsed_ = ReadTimer();
  // checkpoint_ = elapsed_ - checkpoint_;
  // std::cout << "CuDot Time (sec):\t " << checkpoint_ << std::endl;
  //  std::cout << "d1 init " << delta_1 << std::endl;

  // solve for the next conjugate vector until tolerance is satisfied
  while (delta_1 > tolerance && k < maxActive) {
    // q = Ap
    cudaSafeCall(cudaMemcpy(d_scalar1, &scale_1, sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_scalar2, &scale_2, sizeof(float), cudaMemcpyHostToDevice));
    //cublasSafeCall(cublasSsymv(*handle, CUBLAS_FILL_MODE_UPPER, maxActive, d_scalar1, activeSetBuffers->active_kernel_matrix, maxActive, d_p, 1, d_scalar2, d_q, 1));

    cublasSafeCall(cublasSgemv(*handle, CUBLAS_OP_N, maxActive, numActive, d_scalar1, activeSetBuffers->active_kernel_matrix, maxActive, d_p, 1, d_scalar2, d_q, 1));

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "Gemv Time (sec):\t " << checkpoint_ << std::endl;
  
    // float hostK;
    // cudaSafeCall(cudaMemcpy(&hostK, activeSetBuffers->active_kernel_matrix, sizeof(float), cudaMemcpyDeviceToHost));
    // float hostT1;
    // cudaSafeCall(cudaMemcpy(&hostT1, activeSetBuffers->active_targets, sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "K " << hostK << std::endl;
    // std::cout << "Target " << hostT1 << std::endl;
    // float hostP;
    // cudaSafeCall(cudaMemcpy(&hostP, d_p, sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "P " << hostP << std::endl;
    // float hostQ;
    // cudaSafeCall(cudaMemcpy(&hostQ, d_q, sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "Q " << hostQ << std::endl;

    // s = p^T q 
    cublasSafeCall(cublasSdot(*handle, numActive, d_p, 1, d_q, 1, &(d_scalar1[0])));
    cudaSafeCall(cudaMemcpy(&s, d_scalar1, sizeof(float), cudaMemcpyDeviceToHost));

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "PQ Dot Time (sec):\t " << checkpoint_ << std::endl;

    t = delta_1 / s;
    // std::cout << "s " << s << std::endl;
    // std::cout << "t " << t << std::endl;

    // alpha = alpha + t * p
    cudaSafeCall(cudaMemcpy(d_scalar1, &t, sizeof(float), cudaMemcpyHostToDevice));
    cublasSafeCall(cublasSaxpy(*handle, numActive, d_scalar1, d_p, 1, d_alpha, 1)); 

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "P Alpha Inc Time (sec):\t " << checkpoint_ << std::endl;

    // r = r - t * q
    t = -1 * t;
    cudaSafeCall(cudaMemcpy(d_scalar1, &t, sizeof(float), cudaMemcpyHostToDevice));
    cublasSafeCall(cublasSaxpy(*handle, numActive, d_scalar1, d_q, 1, d_r, 1)); 

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "R update Time (sec):\t " << checkpoint_ << std::endl;

    // delta_1 = r^T r
    delta_0 = delta_1;
    cublasSafeCall(cublasSdot(*handle, numActive, d_r, 1, d_r, 1, &(d_scalar1[0])));
    cudaSafeCall(cudaMemcpy(&delta_1, d_scalar1, sizeof(float), cudaMemcpyDeviceToHost));

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "R Norm Time (sec):\t " << checkpoint_ << std::endl;

    // p = r + beta * p
    s = delta_0 / delta_1;
    cudaSafeCall(cudaMemcpy(d_scalar1, &s, sizeof(float), cudaMemcpyHostToDevice));
    cublasSafeCall(cublasSaxpy(*handle, numActive, d_scalar1, d_r, 1, d_p, 1));

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "Conj Vec Time (sec):\t " << checkpoint_ << std::endl;

    s = 1.0f / s;
    cudaSafeCall(cudaMemcpy(d_scalar1, &s, sizeof(float), cudaMemcpyHostToDevice));
    cublasSafeCall(cublasSscal(*handle, numActive, d_scalar1, d_p, 1));

    // checkpoint_ = elapsed_;
    // elapsed_ = ReadTimer();
    // checkpoint_ = elapsed_ - checkpoint_;
    // std::cout << "Scale p Time (sec):\t " << checkpoint_ << std::endl;
    
    k++;
  }

  // float hostMu;
  // cudaSafeCall(cudaMemcpy(&hostMu, &(d_alpha[0]), sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "Solve Linear Alpha " << hostMu << std::endl;

  return true;
}

bool GpuActiveSetSelector::GpPredictChol(MaxSubsetBuffers* subsetBuffers,
					 ActiveSetBuffers* activeSetBuffers, int index,
					 GaussianProcessHyperparams hypers, float* d_kernelVector,
					 float* d_L, float* d_alpha, float* d_gamma,
					 cublasHandle_t* handle, float* d_mu, float* d_sigma)
{
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;
  int numPts = subsetBuffers->num_pts;

  // compute the kernel vector
  compute_kernel_vector(activeSetBuffers, subsetBuffers, index, d_kernelVector, hypers);

  // solve triangular system
  cudaSafeCall(cudaMemcpy(d_gamma, d_kernelVector, maxActive * sizeof(float), cudaMemcpyDeviceToDevice));

  // float hostKV;
  // cudaSafeCall(cudaMemcpy(&hostKV, d_gamma, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "KV: " << hostKV << std::endl;

  cublasSafeCall(cublasStrsv(*handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
			     numActive, d_L, maxActive, d_gamma, 1));

  // float hostG;
  // cudaSafeCall(cudaMemcpy(&hostG, d_gamma, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "G: " << hostG << std::endl;
    
  // dot product to get the resulting mean and variance reduction
  cublasSafeCall(cublasSdot(*handle, numActive, d_alpha, 1, d_kernelVector, 1, &(d_mu[index])));
  cublasSafeCall(cublasSdot(*handle, numActive, d_gamma, 1, d_gamma, 1, &(d_sigma[index])));

  float hostSig;
  cudaSafeCall(cudaMemcpy(&hostSig, &(d_sigma[index]), sizeof(float), cudaMemcpyDeviceToHost));
  VLOG(1) << "Sig at " << index << " : " << hostSig << std::endl;

  return true;
}

bool GpuActiveSetSelector::GpPredictCholBatch(MaxSubsetBuffers* subsetBuffers, ActiveSetBuffers* activeSetBuffers,
					      int index, int batchSize, GaussianProcessHyperparams hypers,
					      float* d_kernelVectors, float* d_L, float* d_alpha, float* d_gamma,
					      float* d_scalar1, float* d_scalar2, cublasHandle_t* handle, float* d_mu, float* d_sigma)
{
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;
  int numPts = subsetBuffers->num_pts;

  // float hostL;
  // cudaSafeCall(cudaMemcpy(&hostL, d_L, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "Pred L before: " << hostL << std::endl;

  // compute the kernel vector
  compute_kernel_vector_batch(activeSetBuffers, subsetBuffers, index, batchSize, d_kernelVectors, hypers);

  // solve triangular system
  cudaSafeCall(cudaMemcpy(d_gamma, d_kernelVectors, maxActive * batchSize * sizeof(float), cudaMemcpyDeviceToDevice));

  float hostKV;
  if (index == 318 || index == 319 || index == 478 || index == 615) {
    cudaSafeCall(cudaMemcpy(&hostKV, d_gamma, sizeof(float), cudaMemcpyDeviceToHost));
    VLOG(2) << "KV for index " << index << ": " << hostKV;
  }
  // cudaSafeCall(cudaMemcpy(&hostL, d_L, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "Pred L after: " << hostL << std::endl;

  cublasSafeCall(cublasStrsm(*handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
			     CUBLAS_DIAG_NON_UNIT, numActive, batchSize, d_scalar1, d_L,
			     maxActive, d_gamma, maxActive));

  float hostG;
  if (index == 318 || index == 319 || index == 478 || index == 615) {
    cudaSafeCall(cudaMemcpy(&hostG, d_gamma, sizeof(float), cudaMemcpyDeviceToHost));
    VLOG(2) << "V for index " << index << ": " << hostG;
  }
    
  // dot product to get the resulting mean and variance reduction
  //  std::cout << "Batch Mult" << std::endl;
  cublasSafeCall(cublasSgemv(*handle, CUBLAS_OP_T, numActive, batchSize, d_scalar1, d_kernelVectors, maxActive, d_alpha,
                             1, d_scalar2, d_mu + index, 1));
  float hostMean;
  if (index == 318 || index == 319 || index == 478 || index == 615) {
    cudaSafeCall(cudaMemcpy(&hostMean, d_mu + index, sizeof(float), cudaMemcpyDeviceToHost));
    VLOG(2) << "Mean for index " << index << ": " << hostMean;
  }

  // get the variance
  //  std::cout << "Batch norm" << std::endl;
  norm_columns(d_gamma, d_sigma + index, numActive, batchSize, maxActive);

  float hostSig;
  if (index == 318 || index == 319 || index == 478 || index == 615) {
    cudaSafeCall(cudaMemcpy(&hostSig, d_sigma + index, sizeof(float), cudaMemcpyDeviceToHost));
    VLOG(2) << "Var for index " << index << ": " << hostSig;
  }
  return true;
}

bool GpuActiveSetSelector::SolveLinearSystemChol(ActiveSetBuffers* activeSetBuffers,
						 float* target, float* d_L,
						 float* d_alpha, cublasHandle_t* handle)
{
  // parallel cholesky decomposition
  int numActive = activeSetBuffers->num_active;
  int maxActive = activeSetBuffers->max_active;

  // perform chol decomp to solve using upper decomp
  cudaSafeCall(cudaMemcpy(d_L, activeSetBuffers->active_kernel_matrix, maxActive * maxActive * sizeof(float), cudaMemcpyDeviceToDevice));
  cudaSafeCall(cudaMemcpy(d_alpha, target, maxActive * sizeof(float), cudaMemcpyDeviceToDevice));

  //  std::cout << "Before chol " << numActive << " " << maxActive << std::endl;
  culaSafeCall(culaDeviceSpotrf('U', numActive, d_L, maxActive));
  //  std::cout << "After chol " << numActive << std::endl;
  culaSafeCall(culaDeviceSpotrs('U', numActive, 1, d_L, maxActive, d_alpha, maxActive));

  // float hostA;
  // cudaSafeCall(cudaMemcpy(&hostA, d_alpha, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "Alpha: " << hostA << std::endl;
  // float hostL;
  // cudaSafeCall(cudaMemcpy(&hostL, d_L, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "L: " << hostL << std::endl;

  return true;
}

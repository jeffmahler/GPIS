#include "cuda_macros.h"
#include "classification_buffers.h"

extern "C" void construct_classification_buffers(ClassificationBuffers *buffers, int num_pts) {
  // assign params
  buffers->num_pts = num_pts;
  
  // allocate buffers
  cudaSafeCall(cudaMalloc((void**)buffers->above, num_pts * sizeof(unsigned char)));
  cudaSafeCall(cudaMalloc((void**)buffers->below, num_pts * sizeof(unsigned char)));

  // set all to 0 (all points are initially undetermined
  cudaSafeCall(cudaMemset((void**)buffers->above, 0, num_pts * sizeof(unsigned char)));  
  cudaSafeCall(cudaMemset((void**)buffers->below, 0, num_pts * sizeof(unsigned char)));  
}

extern "C" void free_classification_buffers(ClassificationBuffers *buffers) {
  // free everything
  cudaSafeCall(cudaFree(buffers->above));
  cudaSafeCall(cudaFree(buffers->below));
}


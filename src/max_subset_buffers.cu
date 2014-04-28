#include "cuda_macros.h"
#include "max_subset_buffers.h"

extern "C" void construct_max_subset_buffers(MaxSubsetBuffers *buffers, float* input_points, float* target_points, int dim_input, int dim_target, int num_pts) {
  // assign params
  buffers->dim_input = dim_input;
  buffers->dim_target = dim_target;
  buffers->num_pts = num_pts;
  
  // allocate buffers
  cudaSafeCall(cudaMalloc((void**)buffers->inputs, dim_input * num_pts * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)buffers->targets, dim_target * num_pts * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)buffers->active, num_pts * sizeof(unsigned char)));
  cudaSafeCall(cudaMalloc((void**)buffers->scores, num_pts * sizeof(float)));

  // set buffs
  cudaSafeCall(cudaMemcpy((void**)(buffers->inputs), input_points, dim_input * num_pts * sizeof(float), cudaMemcpyHostToDevice));  
  cudaSafeCall(cudaMemcpy((void**)(buffers->targets), target_points, dim_target * num_pts * sizeof(float), cudaMemcpyHostToDevice));  

  // set all active to 0 initially
  cudaSafeCall(cudaMemset((void**)buffers->active, 0, num_pts * sizeof(unsigned char)));  
}

extern "C" void activate_max_subset_buffers(MaxSubsetBuffers* buffers, int index) {
  cudaSafeCall(cudaMemset((void**)(buffers->active + index), 1, sizeof(unsigned char)));
}

extern "C" void free_max_subset_buffers(MaxSubsetBuffers *buffers) {
  // free everything
  cudaSafeCall(cudaFree(buffers->inputs));
  cudaSafeCall(cudaFree(buffers->targets));
  cudaSafeCall(cudaFree(buffers->active));
  cudaSafeCall(cudaFree(buffers->scores));
}


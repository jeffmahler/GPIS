#include "cuda_macros.h"
#include "active_set_buffers.h"

extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, float* active_inputs, float* active_targets, int dim_input, int dim_target, int max_active) {
  // assign params
  buffers->max_active = max_active;
  buffers->num_active = 0;
  
  // allocate buffers
  cudaSafeCall(cudaMalloc((void**)buffers->active_inputs, dim_input * max_active * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)buffers->active_targets, dim_target * max_active * sizeof(float)));
}

extern "C" void free_active_set_buffers(ActiveSetBuffers *buffers) {
  // free everything
  cudaSafeCall(cudaFree(buffers->active_inputs));
  cudaSafeCall(cudaFree(buffers->active_targets));
}


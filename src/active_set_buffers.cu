#include "cuda_macros.h"
#include "active_set_buffers.h"

#define BLOCK_DIM_X 128
#define GRID_DIM_X 64

#define MAT_IJ_TO_LINEAR(i, j, dim) ((i) + (j)*(dim))

extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, int dim_input, int dim_target, int max_active) {
  // assign params
  buffers->max_active = max_active;
  buffers->num_active = 0;
  buffers->dim_input = dim_input;
  buffers->dim_target = dim_target;  

  // allocate buffers
  cudaSafeCall(cudaMalloc((void**)&(buffers->active_inputs), dim_input * max_active * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->active_targets), dim_target * max_active * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->active_kernel_matrix), max_active * max_active * sizeof(float)));

  // set kernel matrix to all zeros
  cudaSafeCall(cudaMemset(buffers->active_targets, 0, max_active * sizeof(float)));
  cudaSafeCall(cudaMemset(buffers->active_kernel_matrix, 0, max_active * max_active * sizeof(float)));
}

extern "C" void free_active_set_buffers(ActiveSetBuffers *buffers) {
  // free everything
  cudaSafeCall(cudaFree(buffers->active_inputs));
  cudaSafeCall(cudaFree(buffers->active_targets));
  cudaSafeCall(cudaFree(buffers->active_kernel_matrix));
}

__device__ float exponential_kernel(float* x, float* y, int dim, int sigma)
{
  float sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += __fmul_rn(__fadd_rn(x[i], -y[i]), __fadd_rn(x[i], -y[i]));
    //    printf("sum %f\n", sum);
  }
  return __expf(-sum / (2 * sigma));
}

__global__ void compute_kernel_vector_kernel(float* active_inputs, float* all_inputs, float* kernel_vector, int index, float sigma, int dim_input, int num_pts, int num_active, int max_active)
{
  float local_new_input[MAX_DIM_INPUT];
  float local_active_input[MAX_DIM_INPUT];

  int global_x = threadIdx.x + blockDim.x * blockIdx.x;
  float kernel_val = 0.0f;

  if (global_x >= max_active)
    return;

  // float test = all_inputs[1];
  // if (threadIdx.x == 0 && blockIdx.x == 0)
  //   printf("Test kernel %f\n", test);

  __syncthreads();
  if (global_x < num_active) {
    // read new input into local memory
    for (int i = 0; i < dim_input; i++) {
      local_new_input[i] = all_inputs[index + i*num_pts];
      //      printf("KV New %d %d %f \n", i, index, local_new_input[i]);
    }
    // coalesced read of active input to compute kernel with
    for (int i = 0; i < dim_input; i++) {
      local_active_input[i] = active_inputs[global_x + i*num_pts];
      //      printf("Active %d %d %f \n", i, global_x, local_active_input[i]);
    }

    kernel_val = exponential_kernel(local_new_input, local_active_input, dim_input, sigma);
    //    printf("Kernel val %d %f\n", index, kernel_val/*, local_new_input[0], local_new_input[1], local_active_input[0], local_active_input[1]*/);
  }

  // coalesced value write to vector
  __syncthreads();
  kernel_vector[global_x] = kernel_val;
}

extern "C" void compute_kernel_vector(ActiveSetBuffers *active_buffers, MaxSubsetBuffers *subset_buffers, int index, float* kernel_vector, GaussianProcessHyperparams hypers)
{
  dim3 block_dim(BLOCK_DIM_X, 1, 1);
  dim3 grid_dim(ceilf((float)(active_buffers->num_active)/(float)(block_dim.x)), 1, 1);

  cudaSafeCall((compute_kernel_vector_kernel<<<grid_dim, block_dim>>>(active_buffers->active_inputs, subset_buffers->inputs, kernel_vector, index, hypers.sigma, active_buffers->dim_input, subset_buffers->num_pts, active_buffers->num_active, active_buffers->max_active)));
}

__global__ void update_kernel_matrix_kernel(float* kernel_matrix, float* active_inputs, float* active_targets, float* all_inputs, float* all_targets, float beta, float sigma, int index, int dim_input, int dim_target, int num_pts, int num_active, int max_active)
{
  // parameters
  __shared__ int segment_size;

  float local_new_input[MAX_DIM_INPUT];
  float local_active_input[MAX_DIM_INPUT];
  float local_new_target[MAX_DIM_INPUT];

  // read global variables into shared memory
  if (threadIdx.x == 0) {
    segment_size = max((int)ceilf((float)(num_active+1)/(float)GRID_DIM_X), 1);
  }

  int global_x = 0;
  float kernel = 0.0f;

  __syncthreads();
  for (int i = 0; i * blockDim.x < segment_size; i++) {
    global_x = threadIdx.x + i * blockDim.x + segment_size * blockIdx.x;

    // fetch new data from global menory
    for (int j = 0; j < dim_input; j++) {
      local_new_input[j] = all_inputs[index + j*num_pts];
    } 
    for (int j = 0; j < dim_target; j++) {
      local_new_target[j] = all_targets[index + j*num_pts];
    }

    // fetch active points from global memory
    if (global_x < segment_size * (blockIdx.x + 1) && global_x < num_active) {    
      for (int j = 0; j < dim_input; j++) {
  	local_active_input[j] = active_inputs[global_x + j*num_pts];
      }
      
      kernel = exponential_kernel(local_new_input, local_active_input, dim_input, sigma);
    }

    // coalesced write to new column and row
    __syncthreads();
    if (global_x < segment_size * (blockIdx.x + 1) && global_x < num_active+1) {
      kernel_matrix[MAT_IJ_TO_LINEAR(global_x, num_active, max_active)] = kernel;
      kernel_matrix[MAT_IJ_TO_LINEAR(num_active, global_x, max_active)] = kernel;   
    }

    // coalesced write to active inputs
    __syncthreads();
    if (i == 0 && global_x < dim_input && global_x < segment_size * (blockIdx.x + 1)) {
      active_inputs[num_active + global_x*num_pts] = local_new_input[global_x];
      //      printf("new input %d %f\n", global_x, local_new_input[global_x]);
    }
      
    // coalesced write to active targets
    __syncthreads();
    if (i == 0 && global_x < dim_target && global_x < segment_size * (blockIdx.x + 1)) {
      active_targets[num_active + global_x*num_pts] = local_new_target[global_x];
      //      printf("new target %d %f\n", global_x, local_new_target[global_x]);
    }
      
    // write diagonal term
    __syncthreads();
    if (i == 0 && global_x == 0) {
      float diag_val = exponential_kernel(local_new_input, local_new_input, dim_input, sigma);
      kernel_matrix[MAT_IJ_TO_LINEAR(num_active, num_active, max_active)] = diag_val + beta;
      //      printf("new diag %d %f\n", global_x, kernel_matrix[MAT_IJ_TO_LINEAR(num_active, num_active, max_active)]);
    }
    __syncthreads();
  }
}

extern "C" void update_active_set_buffers(ActiveSetBuffers *active_buffers, MaxSubsetBuffers *subset_buffers, int index, GaussianProcessHyperparams hypers) {

  int dim_input = subset_buffers->dim_input;
  int dim_target = subset_buffers->dim_target;
  if (dim_input > MAX_DIM_INPUT) {
    printf("Error: Input is too high dimensional for update. Aborting...");
    return;
  }
  if (dim_target > MAX_DIM_INPUT) {
    printf("Error: Target is too high dimensional for update. Aborting...");
    return;
  }

  dim3 block_dim(BLOCK_DIM_X, 1, 1);
  dim3 grid_dim(GRID_DIM_X, 1, 1);

  cudaSafeCall((update_kernel_matrix_kernel<<<grid_dim, block_dim>>>(active_buffers->active_kernel_matrix,
								     active_buffers->active_inputs,
								     active_buffers->active_targets,
								     subset_buffers->inputs,
								     subset_buffers->targets,
								     hypers.beta, hypers.sigma,
								     index, dim_input, dim_target,
								     subset_buffers->num_pts,
								     active_buffers->num_active,
								     active_buffers->max_active))); 
  active_buffers->num_active++;
}


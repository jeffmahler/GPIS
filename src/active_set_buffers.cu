#include "cuda_macros.h"
#include "active_set_buffers.h"

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

#define MAT_IJ_TO_LINEAR(i, j, dim) ((i) + (j)*(dim))

extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, int dim_input, int dim_target, int max_active) {
  // assign params
  buffers->max_active = max_active;
  buffers->num_active = 0;
  buffers->dim_input = dim_input;
  buffers->dim_target = dim_target;  

  // allocate buffers
  cudaSafeCall(cudaMalloc((void**)buffers->active_inputs, dim_input * max_active * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)buffers->active_targets, dim_target * max_active * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)buffers->active_kernel_matrix, max_active * max_active * sizeof(float)));

  // set kernel matrix to all zeros
  cudaSafeCall(cudaMemset(buffers->active_kernel_matrix, 0, max_active * max_active * sizeof(float)));
}

extern "C" void update_active_set_buffers_cpu(ActiveSetBuffers *buffers, float* input_points, float* target_points, int index, GaussianProcessHyperparams hypers) {
  cudaSafeCall(cudaMemcpy(buffers->active_inputs + POINT_INDEX(buffers->num_active, buffers->dim_input), input_points + POINT_INDEX(index, buffers->dim_input), buffers->dim_input * sizeof(float), cudaMemcpyDeviceToDevice)); 
  cudaSafeCall(cudaMemcpy(buffers->active_targets + POINT_INDEX(buffers->num_active, buffers->dim_target), target_points + POINT_INDEX(index, buffers->dim_target), buffers->dim_target * sizeof(float), cudaMemcpyDeviceToDevice)); 
 
  update_kernel_matrix(buffers->active_kernel_matrix, buffers->active_inputs, input_points + POINT_INDEX(index, buffers->dim_input), hypers.beta, hypers.sigma, buffers->dim_input, buffers->num_active, buffers->max_active);

  buffers->num_active++;
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
  }
  return __expf(-sum / (2 * sigma));
}

__global__ void compute_kernel_vector_kernel(float* active_inputs, float* input_point, float* kernel_vector, float sigma, int dim_input, int num_active, int max_active)
{
  __shared__ int s_dim_input;
  __shared__ int s_num_active;
  __shared__ int s_max_active;
  __shared__ float s_sigma;

  float local_input_point[MAX_DIM_INPUT];
  float local_active_input[MAX_DIM_INPUT];

  // read global variables into shared memory
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    s_dim_input = dim_input;
    s_num_active = num_active;
    s_max_active= max_active;
    s_sigma = sigma;
  }
  __syncthreads();

  int global_pos = threadIdx.x + blockDim.x * blockIdx.x;
  float kernel_val = 0.0f;

  if (global_pos >= s_max_active)
    return;

  if (global_pos < s_num_active) {
    // read new input into local memory
    for (int i = 0; i < s_dim_input; i++) {
      local_input_point[i] = input_point[i];
    } 
    
    // coalesced read of active input to compute kernel with
    for (int i = 0; i < s_dim_input; i++) {
      local_active_input[i] = active_inputs[i * s_dim_input + threadIdx.x];
    }

    kernel_val = exponential_kernel(local_input_point, local_active_input, s_dim_input, s_sigma);
  }

  // coalesced value write to vector
  __syncthreads();
  kernel_vector[global_pos] = kernel_val;
}

extern "C" void compute_kernel_vector(ActiveSetBuffers *buffers, float* input_point, float* kernel_vector, GaussianProcessHyperparams hypers)
{
  dim3 block_dim(BLOCK_DIM_X, 1, 1);
  dim3 grid_dim(ceilf((float)(buffers->num_active)/(float)(block_dim.x)), 1, 1);

  cudaSafeCall((compute_kernel_vector_kernel<<<grid_dim, block_dim>>>(buffers->active_inputs, input_point, kernel_vector, hypers.sigma, buffers->dim_input, buffers->num_active, buffers->max_active)));
}

__global__ void update_kernel_matrix_kernel(float* kernel_matrix, float* active_inputs, float* new_input, float beta, float sigma, int dim_input, int num_active, int max_active)
{
  __shared__ int s_dim_input;
  __shared__ int s_num_active;
  __shared__ int s_max_active;
  __shared__ float s_beta;
  __shared__ float s_sigma;

  float local_new_input[MAX_DIM_INPUT];
  float local_active_input[MAX_DIM_INPUT];

  // read global variables into shared memory
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    s_dim_input = dim_input;
    s_num_active = num_active;
    s_max_active = max_active;
    s_beta = beta;
    s_sigma = sigma;
  }
  __syncthreads();

  int global_pos = threadIdx.x + blockDim.x * blockIdx.x;
  float kernel_val = 0.0f;

  if (global_pos >= s_max_active)
    return;

  if (global_pos < num_active) {
    // read new input into local memory
    for (int i = 0; i < s_dim_input; i++) {
      local_new_input[i] = new_input[i];
    } 
    
    // coalesced read of active input to compute kernel with
    for (int i = 0; i < s_dim_input; i++) {
      local_active_input[i] = active_inputs[i * s_dim_input + threadIdx.x];
    }

    kernel_val = exponential_kernel(local_new_input, local_active_input, s_dim_input, s_sigma);
  }

  // coalesced value write to new column
  __syncthreads();
  kernel_matrix[MAT_IJ_TO_LINEAR(global_pos, s_num_active+1, s_max_active)] = kernel_val;

  // (coalesced?) value write to new row
  __syncthreads();
  kernel_matrix[MAT_IJ_TO_LINEAR(s_num_active+1, global_pos, s_max_active)] = kernel_val;

  // write diagonal term
  if (global_pos == 0) {
    float diag_val = exponential_kernel(local_new_input, local_new_input, s_dim_input, s_sigma);
    kernel_matrix[MAT_IJ_TO_LINEAR(s_num_active+1, s_num_active+1, s_max_active)] = diag_val + s_beta;
  }
}

// private functions and CUDA kernel calls
extern "C" void update_kernel_matrix(float* kernel_matrix, float* active_inputs, float* new_input, float beta, float sigma, int dim_input, int num_active, int max_active) {
  if (dim_input > MAX_DIM_INPUT) {
    printf("Error: Input is too high dimensional for update. Aborting...");
    return;
  }

  dim3 block_dim(BLOCK_DIM_X, 1, 1);
  dim3 grid_dim(ceilf((float)(num_active)/(float)(block_dim.x)), 1, 1);

  cudaSafeCall((update_kernel_matrix_kernel<<<grid_dim, block_dim>>>(kernel_matrix, active_inputs, new_input, beta, sigma, dim_input, num_active, max_active))); 
}

#include "cuda_macros.h"
#include "max_subset_buffers.h"

#include <math.h>

#define BLOCK_DIM_X 128
#define GRID_DIM_X  128
#define INIT_SCORE -1e6
#define NO_INDEX   -1

extern "C" void construct_max_subset_buffers(MaxSubsetBuffers *buffers, float* input_points, float* target_points, int dim_input, int dim_target, int num_pts) {
  // assign params
  buffers->dim_input = dim_input;
  buffers->dim_target = dim_target;
  buffers->num_pts = num_pts;
  
  // allocate buffers
  cudaSafeCall(cudaMalloc((void**)&(buffers->inputs), dim_input * num_pts * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->targets), dim_target * num_pts * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->active), num_pts * sizeof(unsigned char)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->scores), GRID_DIM_X * sizeof(float)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->indices), GRID_DIM_X * sizeof(int)));
  cudaSafeCall(cudaMalloc((void**)&(buffers->d_next_index), sizeof(int)));  

  // set buffs
  cudaSafeCall(cudaMemcpy(buffers->inputs, input_points, dim_input * num_pts * sizeof(float), cudaMemcpyHostToDevice));  
  cudaSafeCall(cudaMemcpy(buffers->targets, target_points, dim_target * num_pts * sizeof(float), cudaMemcpyHostToDevice));  

  // set all active to 0 initially
  cudaSafeCall(cudaMemset(buffers->active, 0, num_pts * sizeof(unsigned char)));  
}

extern "C" void activate_max_subset_buffers(MaxSubsetBuffers* buffers, int index) {
  cudaSafeCall(cudaMemset(buffers->active + index, 1, sizeof(unsigned char)));
  cudaSafeCall(cudaMemcpy(buffers->d_next_index, &index, sizeof(int), cudaMemcpyHostToDevice));
}

extern "C" void free_max_subset_buffers(MaxSubsetBuffers *buffers) {
  // free everything
  cudaSafeCall(cudaFree(buffers->inputs));
  cudaSafeCall(cudaFree(buffers->targets));
  cudaSafeCall(cudaFree(buffers->active));
  cudaSafeCall(cudaFree(buffers->scores));
  cudaSafeCall(cudaFree(buffers->indices));
  cudaSafeCall(cudaFree(buffers->d_next_index));
}

__device__ float subset_exponential_kernel(float* x, float* y, int dim, int sigma)
{
  float sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += __fmul_rn(__fadd_rn(x[i], -y[i]), __fadd_rn(x[i], -y[i]));
  }
  return __expf(-sum / (2 * sigma));
}

__global__ void distributed_point_evaluation_kernel(float* inputs, float* scores, int* indices,
						    unsigned char* active, unsigned char* upper,
						    unsigned char* lower, float* mean,
						    float* variance, float level,
						    float var_scaling, float eps, float beta, float sigma,
						    int dim_input, int num_pts)
{
  // max score for each thread
  __shared__ float s_scores[BLOCK_DIM_X];
  __shared__ int   s_indices[BLOCK_DIM_X];

  // parameters
  __shared__ int segment_size;
  __shared__ int s_eps;

  // allocate local computation buffers
  float point[MAX_DIM_INPUT];
  float pred_mean = 0.0f;
  float pred_var = 0.0f;
  float ambiguity = 0.0f;
  float kernel = 0.0f;
  unsigned char active_flag = 0;
  unsigned char upper_flag = 0;
  unsigned char lower_flag = 0;

  // initialize
  if (threadIdx.x == 0) {
    segment_size = (int)ceilf((float)num_pts/(float)GRID_DIM_X);
    s_eps = eps;
  }

  // initialize scores and count
  s_scores[threadIdx.x] = INIT_SCORE;
  s_indices[threadIdx.x] = NO_INDEX;
  __syncthreads();
  
  // position
  int global_x = 0;

  // loop over points
  for (int i = 0; i * BLOCK_DIM_X < segment_size; i++) {
    global_x = threadIdx.x + i * BLOCK_DIM_X + segment_size * blockIdx.x;

    // fetch point from global memory
    if (global_x < segment_size * (blockIdx.x + 1) && global_x < num_pts) {
      for (int j = 0; j < dim_input; j++) {
  	point[j] = inputs[global_x + j * num_pts];
      }
      pred_mean = mean[global_x];
      pred_var = variance[global_x];
      active_flag = active[global_x];
      upper_flag = upper[global_x];
      lower_flag = lower[global_x];
    }

    if (global_x < segment_size * (blockIdx.x + 1) && global_x < num_pts) {
      // compute things only if we do not know this point yet
      if (!active_flag && !upper_flag && !lower_flag) { 
  	// compute the ambiguity (see Gotovos et al for more info)
  	kernel = subset_exponential_kernel(point, point, dim_input, sigma);
        kernel += beta;
        // if (point[0] == 23 && point[1] == 8) {
        //   printf("Point %f %f kernel: %f var red: %f\n", point[0], point[1], kernel, pred_var);
        // }
  	pred_var = kernel - pred_var;
  	pred_var = var_scaling * pred_var;

  	// check upper, lower
  	if (pred_mean + pred_var - s_eps <= level) {
          lower_flag = 1;
        }

        if (pred_mean - pred_var + s_eps > level) {
          upper_flag = 1;
        }

  	// update local ambiguity score
  	ambiguity = min(pred_mean + pred_var - level, level - (pred_mean - pred_var)); 

        // if (global_x == 318 || global_x == 319 || global_x == 478 || global_x == 615) {
        //   printf("Index %d score: %f flags: %d %d %d\n", global_x, ambiguity, active_flag, upper_flag, lower_flag);
        // }
        // if (point[0] == 23 && point[1] == 8) {
        //   printf("Point %f %f index: %d ambiguity: %f mean: %f std: %f\n", point[0], point[1], global_x, ambiguity, pred_mean, pred_var);
        // }

  	if (ambiguity > s_scores[threadIdx.x]) {
  	  s_scores[threadIdx.x] = ambiguity;
  	  s_indices[threadIdx.x] = global_x;
  	}
      }
    }
    // write upper / lower flags
    __syncthreads();
    if (global_x < segment_size * (blockIdx.x + 1) && global_x < num_pts) {    
      lower[global_x] = lower_flag;
      upper[global_x] = upper_flag;
    }
  }

  // max reduction
  global_x = threadIdx.x;
  for (unsigned int stride = BLOCK_DIM_X >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (global_x < stride && (global_x + stride) < BLOCK_DIM_X) {
      if (s_scores[global_x + stride] > s_scores[global_x]) {
        s_scores[global_x] = s_scores[global_x + stride];
        s_indices[global_x] = s_indices[global_x + stride];
      }
    }
  }

  // write results to global memory
  __syncthreads();
  scores[blockIdx.x] = s_scores[0];
  indices[blockIdx.x] = s_indices[0];
}

__global__ void distributed_point_reduction_kernel(float* scores, int* indices, unsigned char* active, int* g_index)
{
  //  buffers for shared indices, scores
  __shared__ float s_scores[GRID_DIM_X];
  __shared__ int s_indices[GRID_DIM_X];


  int global_x = threadIdx.x;
  __syncthreads();
  if (global_x < GRID_DIM_X) {
    s_scores[global_x] = scores[global_x];
    s_indices[global_x] = indices[global_x];
  }
  __syncthreads();

  for (unsigned int stride = GRID_DIM_X >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (global_x < stride && (global_x + stride) < BLOCK_DIM_X) {
      if (s_scores[global_x + stride] > s_scores[global_x]) {
        s_scores[global_x] = s_scores[global_x + stride];
        s_indices[global_x] = s_indices[global_x + stride];
      }
    }
  }  

  // write result to global memory
  __syncthreads();
  if (threadIdx.x == 0) {
    scores[0] = s_scores[0];
    g_index[0] = s_indices[0];
    if (s_indices[0] >= 0) {
      active[s_indices[0]] = 1;
    }
    //    printf("Chose %d as next index...\n", g_index[0]);
  }
}

extern "C" void find_best_active_set_candidate(MaxSubsetBuffers* subsetBuffers, ClassificationBuffers* classificationBuffers,
                                               float* d_mu, float* d_sigma, float level, float beta, float eps,
                                               GaussianProcessHyperparams hypers)
{
  float var_scaling = sqrt(beta); // square root to get the right answer
  int num_pts = subsetBuffers->num_pts;
  int dim_input = subsetBuffers->dim_input;

  dim3 block_dim(BLOCK_DIM_X, 1, 1);
  dim3 grid_dim(GRID_DIM_X, 1, 1);

  // distributed ambiguity calculation, classification, and max reduction
  cudaSafeCall((distributed_point_evaluation_kernel<<<grid_dim, block_dim>>>(subsetBuffers->inputs,
                                                                             subsetBuffers->scores,
                                                                             subsetBuffers->indices,
                                                                             subsetBuffers->active,
                                                                             classificationBuffers->upper,
                                                                             classificationBuffers->lower,
                                                                             d_mu, d_sigma,
                                                                             level, var_scaling, eps,
                                                                             hypers.beta,
                                                                             hypers.sigma,
                                                                             dim_input,
                                                                             num_pts)));

  // distributed sum reduction
  cudaSafeCall((distributed_point_reduction_kernel<<<1, grid_dim>>>(subsetBuffers->scores,
  								    subsetBuffers->indices,
								    subsetBuffers->active,
  								    subsetBuffers->d_next_index)));
}

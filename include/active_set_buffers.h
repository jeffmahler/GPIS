// Buffers to hold active set (with current elements at front)
#pragma once

#define POINT_INDEX(x, dim) ((dim)*x)

typedef struct {
  float beta;
  float sigma;
} GaussianProcessHyperparams;

typedef struct {
  float* active_inputs;
  float* active_targets;
  float* active_kernel_matrix;
  int num_active;
  int max_active;
  int dim_input;
  int dim_target;
} ActiveSetBuffers;

// constructor/destructor
extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, int dim_input, int dim_target, int max_active);
extern "C" void update_active_set_buffers_cpu(ActiveSetBuffers *buffers, float* input_points, float* target_points, int index, GaussianProcessHyperparams hypers);
extern "C" void free_active_set_buffers(ActiveSetBuffers *buffers);

// private
extern "C" void update_kernel_matrix(float* kernel_matrix, float* active_inputs, float* new_input, float beta, float sigma, int dim_input, int num_active, int max_active);

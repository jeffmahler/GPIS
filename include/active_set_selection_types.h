// Includes for the active set selection CUDA routines
#pragma once

#define POINT_INDEX(x, dim) ((dim)*(x))
#define IJ_TO_LINEAR(i, j, width) ((i) + (width)*(j))
#define IJK_TO_LINEAR(i, j, k, width, height) ((i) + (width)*(j) + (width)*(height)*(k))

#define MAX_DIM_INPUT 10

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

typedef struct {
  float* inputs;
  float* targets;
  unsigned char* active;
  float* scores; // reduction buffer for scores
  int* indices;  // reduction buffer for indices
  int dim_input;
  int dim_target;
  int num_pts;
} MaxSubsetBuffers;

typedef struct {
  unsigned char* upper;
  unsigned char* lower;
  int num_pts;
} ClassificationBuffers;

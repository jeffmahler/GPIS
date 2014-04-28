// Buffers to hold subset maximizaton items
#pragma once

typedef struct {
  float* inputs;
  float* targets;
  unsigned char* active;
  float* scores;
  int dim_input;
  int dim_target;
  int num_pts;
} MaxSubsetBuffers;

// constructor/destructor
extern "C" void construct_max_subset_buffers(MaxSubsetBuffers *buffers, float* input_points, float* target_points, int dim_input, int dim_target, int num_pts);
extern "C" void activate_max_subset_buffers(MaxSubsetBuffers *buffers, int index);
extern "C" void free_max_subset_buffers(MaxSubsetBuffers *buffers);

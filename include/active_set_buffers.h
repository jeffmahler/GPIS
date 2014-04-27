// Buffers to hold active set (with current elements at front)
#pragma once

typedef struct {
  float* active_inputs;
  float* active_targets;
  int num_active;
  int max_active;
} ActiveSetBuffers;

// constructor/destructor
extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, float* active_inputs, float* active_targets, int dim_input, int dim_target, int max_active);
extern "C" void free_max_subset_buffers(ActiveSetBuffers *buffers);

// Buffers to hold subset maximizaton items
#pragma once

#include "active_set_buffers.h"
#include "classification_buffers.h"

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

// constructor/destructor
extern "C" void construct_max_subset_buffers(MaxSubsetBuffers *buffers, float* input_points, float* target_points, int dim_input, int dim_target, int num_pts);
extern "C" void activate_max_subset_buffers(MaxSubsetBuffers *buffers, int index);
extern "C" void free_max_subset_buffers(MaxSubsetBuffers *buffers);

// compute the ambiguity for each point, reclassify, and choose next point
extern "C" int find_best_active_set_candidate(MaxSubsetBuffers* subsetBuffers, ClassificationBuffers* classificationBuffers, float* d_mu, float* d_sigma, float level, float beta, GaussianProcessHyperparams hypers); 

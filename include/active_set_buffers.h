// Buffers to hold active set (with current elements at front)
#pragma once

#include "active_set_selection_types.h"

// constructor/destructor
extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, int dim_input, int dim_target, int max_active);
extern "C" void free_active_set_buffers(ActiveSetBuffers *buffers);

// helper functions
extern "C" void compute_kernel_vector(ActiveSetBuffers *active_buffers, MaxSubsetBuffers* subset_buffers, int index, float* kernel_vector, GaussianProcessHyperparams hypers);
extern "C" void update_active_set_buffers(ActiveSetBuffers *active_buffers, MaxSubsetBuffers *subset_buffers, int index, GaussianProcessHyperparams hypers);


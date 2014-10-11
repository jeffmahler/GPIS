// Buffers to hold active set (with current elements at front)
#pragma once

#include "active_set_selection_types.h"

// constructor/destructor
extern "C" void construct_active_set_buffers(ActiveSetBuffers *buffers, int dim_input, int dim_target, int max_active);
extern "C" void free_active_set_buffers(ActiveSetBuffers *buffers);

// helper functions
extern "C" void compute_kernel_vector(ActiveSetBuffers *active_buffers, MaxSubsetBuffers* subset_buffers, int index, float* kernel_vector, GaussianProcessHyperparams hypers);
extern "C" void compute_kernel_vector_batch(ActiveSetBuffers *active_buffers, MaxSubsetBuffers* subset_buffers, int index, int batch_size, float* kernel_vectors, GaussianProcessHyperparams hypers);
extern "C" void update_active_set_buffers(ActiveSetBuffers *active_buffers, MaxSubsetBuffers *subset_buffers, GaussianProcessHyperparams hypers);

// random reduction function for solving the linear system fast
extern "C" void norm_columns(float* A, float* x, int m, int n, int lda);

// compute variance of a single point from the chol solved V vector
extern "C" void compute_sqrt_var(float* v, float* x, int m, float sigma, float beta, int dim_input);


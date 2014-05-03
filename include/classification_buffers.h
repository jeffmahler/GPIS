// Buffers to hold classification (for 3 class system)
#pragma once

#include "active_set_selection_types.h"

// constructor/destructor
extern "C" void construct_classification_buffers(ClassificationBuffers *buffers, int num_pts);
extern "C" void free_classification_buffers(ClassificationBuffers *buffers);

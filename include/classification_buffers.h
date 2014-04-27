// Buffers to hold classification (for 3 class system)
#pragma once

typedef struct {
  unsigned char* above;
  unsigned char* below;
  int num_pts;
} ClassificationBuffers;

// constructor/destructor
extern "C" void construct_classification_buffers(ClassificationBuffers *buffers, int num_pts);
extern "C" void free_classification_buffers(ClassificationBuffers *buffers);

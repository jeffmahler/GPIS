// Class to encapsulate selection of the active subset

#pragma once

#include <vector>

#include "active_set_buffers.h"

class GpuActiveSetSelector {

  // possible criteria from which to select the active subset
  enum SubsetSelectionMode {
    ENTROPY,
    LEVEL_SET
  };

 public:
  GpuActiveSetSelector() {}
  ~GpuActiveSetSelector() {}

 public:
  // Select an active subset from 
  bool Select(int maxSize, float* inputPoints, float* targetPoints,
	      SubsetSelectionMode mode,
	      GaussianProcessHyperparams hypers,
	      int inputDim, int targetDim, int numPoints, float tolerance,
	      float* activeInputs, float* activeTargets);
};

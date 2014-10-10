#include "csv.hpp"

#include "active_set_selection_types.h"
#include "cuda_macros.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>

bool WriteCsvGpu(const std::string& csvFilename, float* buffer, int width, int height, int lda)
{
  std::ofstream csvFile(csvFilename.c_str());
  std::string delim = ",";
  float* hostBuffer = new float[width * lda];
  cudaSafeCall(cudaMemcpy(hostBuffer, buffer, width * lda * sizeof(float), cudaMemcpyDeviceToHost));

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      csvFile << hostBuffer[j + i*lda];
      if (i < width-1)
	csvFile << delim;
    }
    csvFile << "\n";
  }
  csvFile.close();

  delete [] hostBuffer;
  return true;
}

bool WriteCsv(const std::string& csvFilename, float* buffer, int width, int height, int lda)
{
  std::ofstream csvFile(csvFilename.c_str());
  std::string delim = ",";

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      csvFile << buffer[j + i*lda];
      if (i < width-1)
	csvFile << delim;
    }
    csvFile << "\n";
  }
  csvFile.close();
  return true;
}

bool ReadCsv(const std::string& csvFilename, float* buffer, int width, int height, int depth, bool storeDepth)
{
  std::ifstream csvFile(csvFilename.c_str());
  if (!csvFile.is_open()) {
    LOG(ERROR) << "Failed to open " << csvFilename;
    return false;
  }
  VLOG(0) << "Opened " << csvFilename;

  int maxChars = 10000;
  char lineBuffer[maxChars];

  int j = 0;
  int k = 0;
  float val = 0.0f;
  int numPts = width*height*depth;
  char delim;

  while(!csvFile.eof() && k < depth) {
    csvFile.getline(lineBuffer, maxChars);

    std::stringstream parser(lineBuffer);
    for (int i = 0; i < width; i++) {
      parser >> val;
      if (i < width-1)
	parser >> delim;
      buffer[IJK_TO_LINEAR(i, j, k, width, height)] = val;
      //      std::cout << buffer[IJK_TO_LINEAR(i, j, k, width, height)] << " " << width << " " << height << std::endl;
    }

    // set the next index
    j++;
    if (j >= height) {
      j = 0;
      k++;
    }
  }
  
  csvFile.close();
  return true;
}

bool ReadTsdf(const std::string& csvFilename, float* inputs, float* targets, int width, int height, int depth, bool storeDepth)
{
  std::ifstream csvFile(csvFilename.c_str());
  if (!csvFile.is_open()) {
    LOG(ERROR) << "Failed to open " << csvFilename;
    return false;
  }

 int maxChars = 10000;
  char buffer[maxChars];

  int j = 0;
  int k = 0;
  float val = 0.0f;
  int numPts = width*height*depth;
  char delim;

  while(!csvFile.eof() && k < depth) {
    csvFile.getline(buffer, maxChars);

    std::stringstream parser(buffer);
    for (int i = 0; i < width; i++) {
      parser >> val;
      if (i < width-1)
	parser >> delim;
      inputs[IJK_TO_LINEAR(i, j, k, width, height) + 0 * numPts] = i;
      inputs[IJK_TO_LINEAR(i, j, k, width, height) + 1 * numPts] = j;
      if (storeDepth) {
	inputs[IJK_TO_LINEAR(i, j, k, width, height) + 2 * numPts] = k;
      }
      targets[IJK_TO_LINEAR(i, j, k, width, height) + 0 * numPts] = val;
    }

    // set the next index
    j++;
    if (j >= height) {
      j = 0;
      k++;
    }
  }
  
  csvFile.close();
  return true;
}

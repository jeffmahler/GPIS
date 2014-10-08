
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sstream>

#include <glog/logging.h>

#include "gpu_active_set_selector.hpp"
#include "max_subset_buffers.h"

#define CONFIG_SIZE 8

#define DEFAULT_CSV "test.csv"
#define DEFAULT_SET_SIZE 100
#define DEFAULT_SIGMA 1
#define DEFAULT_BETA 0.1
#define DEFAULT_WIDTH 100
#define DEFAULT_HEIGHT 1
#define DEFAULT_DEPTH 1
#define DEFAULT_BATCH 1
#define DEFAULT_TOLERANCE 0.001

// read in a configuration file
bool readConfig(const std::string& configFilename, std::string& csvFilename, int& setSize, float& sigma, float& beta, int& width, int& height, int& depth, int& batch)
{
  std::ifstream configFile(configFilename.c_str());
  if (!configFile.is_open()) {
    LOG(ERROR) << "Failed to open " << configFilename;
    return false;
  }

  int maxChars = 1000;
  char buffer[maxChars];

  int i;
  for (i = 0; !configFile.eof() && i < CONFIG_SIZE;) {
    configFile.getline(buffer, maxChars);
    std::stringstream parser(buffer);

    // ignore comment lines
    if (buffer[0] != '#') {
      switch(i) {
       case 0:
	parser >> csvFilename;
       case 1:
	parser >> setSize;
       case 2:
	parser >> sigma;
       case 3:
	parser >> beta;
       case 4:
	parser >> width;
       case 5:
	parser >> height;
       case 6:
	parser >> depth;
       case 7:
	parser >> batch;
      }
      i++;
    }
  }

  configFile.close();
  if (i < CONFIG_SIZE) {
    std::cout << "Illegal configuration - too few params" << std::endl;
    return false;
  }

  return true;
}

void printHelp()
{
  std::cout << "Usage: GPIS [config]" << std::endl;
  std::cout << "\t config - name of configuration file" << std::endl;
}

int main(int argc, char* argv[])
{
  //  srand(1000);//time(NULL));

  if (argc < 2) {
    printHelp();
    return 1;
  }

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_v = 1;

  // read args
  std::string configFilename = argv[1];
  std::string csvFilename = DEFAULT_CSV;
  int setSize = DEFAULT_SET_SIZE;
  float sigma = DEFAULT_SIGMA;
  float beta = DEFAULT_BETA;
  int width = DEFAULT_WIDTH;
  int height = DEFAULT_HEIGHT;
  int depth = DEFAULT_DEPTH;
  int batchSize = DEFAULT_BATCH;
  float tolerance = DEFAULT_TOLERANCE;

  bool opened = readConfig(configFilename, csvFilename, setSize, sigma, beta, width, height, depth, batchSize);
  if (!opened) {
    return 1;
  }

  LOG(INFO) << "Using the following GPIS params:";
  LOG(INFO) << "csv:\t\t" << csvFilename;
  LOG(INFO) << "K:\t\t" << setSize;
  LOG(INFO) << "sigma:\t" << sigma;
  LOG(INFO) << "beta:\t\t" << beta;
  LOG(INFO) << "width:\t" << width;
  LOG(INFO) << "height:\t" << height;
  LOG(INFO) << "depth:\t" << depth;
  LOG(INFO) << "batch:\t" << batchSize;

  GpuActiveSetSelector gpuSetSelector;
  gpuSetSelector.SelectFromGrid(csvFilename, setSize, sigma, beta, width, height, depth, batchSize, tolerance);

  return 0;
}


#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sstream>

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
#define DEFAULT_TOLERANCE 0.01

// read in a configuration file
bool readConfig(const std::string& configFilename, std::string& csvFilename, int& setSize, float& sigma, float& beta, int& width, int& height, int& depth, int& batch)
{
  std::ifstream configFile(configFilename.c_str());
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
  srand(1000);//time(NULL));

  if (argc < 2) {
    printHelp();
    return 1;
  }

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

  readConfig(configFilename, csvFilename, setSize, sigma, beta, width, height, depth, batchSize);
  std::cout << "Using the followig GPIS params:" << std::endl;
  std::cout << "csv:\t" << csvFilename << std::endl;
  std::cout << "K:\t" << setSize << std::endl;
  std::cout << "sigma:\t" << sigma << std::endl;
  std::cout << "beta:\t" << beta << std::endl;
  std::cout << "width:\t" << width << std::endl;
  std::cout << "height:\t" << height << std::endl;
  std::cout << "depth:\t" << depth << std::endl;
  std::cout << "batch:\t" << batchSize << std::endl;

  GpuActiveSetSelector gpuSetSelector;
  gpuSetSelector.SelectFromGrid(csvFilename, setSize, sigma, beta, width, height, depth, batchSize, tolerance);

  return 0;
}

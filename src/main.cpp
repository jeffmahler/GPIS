// Right now this file runs tests, experiments on the fast GPIS construction
#include "active_set_selection_types.h"
#include "csv.hpp"
#include "gpu_active_set_selector.hpp"
#include "max_subset_buffers.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sstream>

#include <glog/logging.h>

#define CONFIG_SIZE 10

#define DEFAULT_CSV "test.csv"
#define DEFAULT_SET_SIZE 100
#define DEFAULT_SIGMA 1
#define DEFAULT_BETA 0.1
#define DEFAULT_WIDTH 100
#define DEFAULT_HEIGHT 1
#define DEFAULT_DEPTH 1
#define DEFAULT_BATCH 1
#define DEFAULT_TOLERANCE 0.001
#define DEFAULT_START_INDEX -1
#define DEFAULT_ACCURACY 1e-4

// read in a configuration file
bool readConfig(const std::string& configFilename, std::string& csvFilename, int& setSize, float& sigma, float& beta,
                int& width, int& height, int& depth, int& batch, float& tolerance, int& startIndex, float& accuracy)
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
       case 8:
        parser >> tolerance;
       case 9:
        parser >> startIndex;
       case 10:
        parser >> accuracy;
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

bool runTest(int width, int height, int depth)
{
  int numPoints = width * height * depth;

  std::string truthPointsFilename = "data/test/activePoints.csv";
  float* truthPoints = new float[numPoints];
  ReadCsv(truthPointsFilename, truthPoints, width, height, depth, false);

  std::string testPointsFilename = "inputs.csv";
  float* testPoints = new float[numPoints];
  ReadCsv(testPointsFilename, testPoints, width, height, depth, false);
  bool samePoints = true;
  float truthVal = 0.0f;
  float testVal = 0.0f;

  for(int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      truthVal = truthPoints[IJK_TO_LINEAR(i, j, 0, width, height)];
      testVal = testPoints[IJK_TO_LINEAR(i, j, 0, width, height)]; 
      //      std::cout << truthVal << " " << testVal << std::endl;

      if (abs(truthVal - testVal) > 1e-4) {
        samePoints = false;
      }
    }
  }

  return samePoints;
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
  FLAGS_v = 2;

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
  int startIndex = DEFAULT_START_INDEX;
  float accuracy = DEFAULT_ACCURACY;
  bool storeDepth = false;

  bool opened = readConfig(configFilename, csvFilename, setSize, sigma, beta, width, height, depth,
                           batchSize, tolerance, startIndex, accuracy);
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
  LOG(INFO) << "tolerance:\t" << tolerance;
  LOG(INFO) << "start:\t" << startIndex;
  LOG(INFO) << "accuracy:\t" << accuracy;

  GpuActiveSetSelector gpuSetSelector;
  gpuSetSelector.SelectFromGrid(csvFilename, setSize, sigma, beta, width, height, depth,
                                batchSize, tolerance, accuracy, storeDepth, startIndex);
  bool success = runTest(2, setSize, 1);
  if (!success) {
    LOG(ERROR) << "TEST FAILED!";
  }
  
  return 0;
}

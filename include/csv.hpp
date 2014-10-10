// Functions for reading / writing CSV files

#pragma once

#include <string>
#include <vector>

/**
 * WriteCsv
 * @brief write a csv file with dimension up to width x height x depth from a GPU buffer
 */
bool WriteCsvGpu(const std::string& csvFilename, float* buffer, int width, int height, int lda);

/**
 * WriteCsv
 * @brief write a csv file with dimension up to width x height x depth from a CPU buffer
 */
bool WriteCsv(const std::string& csvFilename, float* buffer, int width, int height, int lda);

/**
 * ReadCsv
 * @brief read a csv file with dimension up to width x height x depth into a buffer
 */
bool ReadCsv(const std::string& csvFilename, float* buffer, int width, int height, int depth, bool storeDepth);

/**
 * ReadTsdf
 * @brief read a csv file of a TSDF with dimension up to width x height x depth into a buffer
 */
bool ReadTsdf(const std::string& csvFilename, float* inputs, float* targets, int width, int height, int depth, bool storeDepth);

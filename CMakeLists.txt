# CMakeLists for GPIS
cmake_minimum_required (VERSION 2.6)
project(GPIS)

# Flags
set (GPIS_LIB_TYPE "SHARED" CACHE STRING "Library type defaults to shared, options are: SHARED STATIC")

# Dependencies
# Boost
find_package (Boost COMPONENTS filesystem system thread REQUIRED)
  include_directories (${Boost_INCLUDE_DIRS})

# CUDA
find_package (CUDA REQUIRED)
  include_directories (${CUDA_INCLUDE_DIRS})
  set (CUDA_NVCC_FLAGS "--gpu-architecture=sm_20;")

set (DEPENDENCY_LIBS ${Boost_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Output flags
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set (CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/lib)
link_directories (${CMAKE_LIBRARY_PATH})
include_directories (${CMAKE_CURRENT_BINARY_DIR})
include_directories (${CMAKE_CURRENT_BINARY_DIR}/src)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
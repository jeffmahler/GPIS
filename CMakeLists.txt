# CMakeLists for GPIS
cmake_minimum_required (VERSION 2.6)
project(GPIS)

# Flags
set (GPIS_LIB_TYPE "SHARED" CACHE STRING "Library type defaults to shared, options are: SHARED STATIC")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -g ")

# Dependencies
# Boost
find_package (Boost COMPONENTS filesystem system thread REQUIRED)
  include_directories (${Boost_INCLUDE_DIRS})

# CUDA
find_package (CUDA REQUIRED)
  include_directories (${CUDA_INCLUDE_DIRS})
  set (CUDA_NVCC_FLAGS "-g --gpu-architecture=sm_20;")

# CULA (hardcoded)
set(CULA_INCLUDE_DIRS "/usr/local/cula/include")
set(CULA_LIBRARIES "/usr/local/cula/lib64/libcula_lapack.so")
include_directories (${CULA_INCLUDE_DIRS})

set (DEPENDENCY_LIBS ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CMAKE_DL_LIBS} ${CULA_LIBRARIES})
#message(${DEPENDENCY_LIBS})

# Output flags
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set (CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/lib)
link_directories (${CMAKE_LIBRARY_PATH})
include_directories (${CMAKE_CURRENT_BINARY_DIR})
include_directories (${CMAKE_CURRENT_BINARY_DIR}/src)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
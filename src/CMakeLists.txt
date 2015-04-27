# Source CMakeLists directory
file (GLOB_RECURSE SOURCES "*.cpp" "*.cu")
file (GLOB_RECURSE MAIN "*main.cpp")
list (REMOVE_ITEM SOURCES ${MAIN})

# create library
#cuda_add_library (${CMAKE_PROJECT_NAME}_Core ${GPIS_LIB_TYPE} ${SOURCES})
#target_link_libraries (${CMAKE_PROJECT_NAME}_Core ${DEPENDENCY_LIBS})

#add_executable(GPIS main.cpp)
#target_link_libraries(GPIS ${CMAKE_PROJECT_NAME}_Core)

add_executable(shot_extractor shot_extractor.cpp)
target_link_libraries(shot_extractor ${FEATURE_DEPENDENCY_LIBS})

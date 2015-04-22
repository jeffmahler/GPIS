cmake_minimum_required(VERSION 2.8)
project (gqsim)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# OS X
set(EXTRA_LIBRARIES "")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_EXE_LINKER_FLAGS "-framework Cocoa -framework IOKit")
endif()

# 3rdparty libraries
add_subdirectory(3rdparty)

# gqsim
file(GLOB gqsim_SRC "src/*.cpp" "include/*.h")
include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
include_directories("include")
include_directories("3rdparty/gear/include")
include_directories("3rdparty/util/include")
include_directories("3rdparty/tinyxml")
include_directories("3rdparty/forceclosure")
add_executable(gqsim ${gqsim_SRC})
target_link_libraries(gqsim util gear tinyxml forceclosure qhull ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${EXTRA_LIBRARIES})


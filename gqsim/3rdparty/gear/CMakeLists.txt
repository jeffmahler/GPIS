file(GLOB gear_SRC "src/*.cpp" "include/*.h")
include_directories("include")
add_library(gear ${gear_SRC})
set_target_properties(gear PROPERTIES OUTPUT_NAME gear)

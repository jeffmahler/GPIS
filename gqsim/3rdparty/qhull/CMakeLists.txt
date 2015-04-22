file(GLOB qhull_files "*.c" "*.h")
add_library(qhull ${qhull_files})
set_target_properties(qhull PROPERTIES OUTPUT_NAME qhull)

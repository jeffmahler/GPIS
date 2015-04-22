file(GLOB tinyxml_SRC "*.cpp" "*.h")
add_library(tinyxml ${tinyxml_SRC})
set_target_properties(tinyxml PROPERTIES OUTPUT_NAME tinyxml)

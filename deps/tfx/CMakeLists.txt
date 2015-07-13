cmake_minimum_required(VERSION 2.8.3)
project(tfx)

find_package(catkin REQUIRED COMPONENTS geometry_msgs)

catkin_package(
   INCLUDE_DIRS include
   CATKIN_DEPENDS geometry_msgs)

catkin_python_setup()

#install(PROGRAMS scripts/tf_echo scripts/save_tf scripts/load_tf scripts/topic_echo scripts/publisher scripts/pose_publisher scripts/tf_publisher
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

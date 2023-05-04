cmake_minimum_required(VERSION 3.0.2)
project(fcc)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf2_ros
  tf2_geometry_msgs
  dynamic_reconfigure
  autoware_msgs
  jsk_recognition_msgs
)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

generate_dynamic_reconfigure_options(
  cfg/fcc.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fcc
  CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/dbscan_kdtree.hpp
# )

add_executable(fcc_node 
  src/fcc.cpp
)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(fcc_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(fcc_node
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
cmake_minimum_required(VERSION 3.16)
project(hand_ncnn_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs highgui calib3d)

option(NCNN_ROOT "" "")
if(NCNN_ROOT)
  set(ncnn_DIR "${NCNN_ROOT}/lib/cmake/ncnn" CACHE PATH "")
endif()
find_package(ncnn QUIET)

add_executable(hand_ncnn_demo main.cpp)
target_include_directories(hand_ncnn_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hand_ncnn_demo PRIVATE ${OpenCV_LIBS})

if(ncnn_FOUND)
  target_include_directories(hand_ncnn_demo PRIVATE ${ncnn_DIR}/../../include/ncnn ${ncnn_DIR}/../../include)
  target_link_libraries(hand_ncnn_demo PRIVATE ncnn)
  target_compile_definitions(hand_ncnn_demo PRIVATE HAS_NCNN=1)
else()
  message(WARNING "ncnn not found — building demo without NCNN inference; set -Dncnn_DIR=")
  target_compile_definitions(hand_ncnn_demo PRIVATE HAS_NCNN=0)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hand_ncnn_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

# ROS2 Dedicated Compilation Target
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(sensor_msgs QUIET)
find_package(cv_bridge QUIET)

if(rclcpp_FOUND AND sensor_msgs_FOUND AND cv_bridge_FOUND)
  message(STATUS "ROS2 Dependencies Located: Building hand_ncnn_ros2 target.")
  add_executable(hand_ncnn_ros2 main_ros2.cpp)
  target_include_directories(hand_ncnn_ros2 PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(hand_ncnn_ros2 ${OpenCV_LIBS})

  ament_target_dependencies(hand_ncnn_ros2 rclcpp sensor_msgs cv_bridge)

  if(ncnn_FOUND)
    target_include_directories(hand_ncnn_ros2 PUBLIC ${ncnn_DIR}/../../include/ncnn ${ncnn_DIR}/../../include)
    target_link_libraries(hand_ncnn_ros2 ncnn)
    target_compile_definitions(hand_ncnn_ros2 PUBLIC HAS_NCNN=1)
  else()
    target_compile_definitions(hand_ncnn_ros2 PUBLIC HAS_NCNN=0)
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(hand_ncnn_ros2 OpenMP::OpenMP_CXX)
  endif()
else()
  message(WARNING "ROS2 (rclcpp, cv_bridge, sensor_msgs) NOT found. Skipping ROS2 stereocam node compilation.")
endif()

cmake_minimum_required(VERSION 3.16)
project(hand_ncnn_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs)

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

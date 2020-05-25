cmake_minimum_required(VERSION 3.5)

project(tf_interfaces)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ImageClassification.srv"
  "srv/ImageDetection.srv"
  DEPENDENCIES std_msgs sensor_msgs vision_msgs
)

ament_package()
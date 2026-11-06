cmake_minimum_required(VERSION 3.16)

project(robottrace_examples)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(pipeline src/pipeline.cpp)
ament_target_dependencies(pipeline rclcpp std_msgs std_srvs)
add_executable(event_cost src/event_cost.cpp)
foreach(target pipeline event_cost)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(${target} PRIVATE include)
    install(TARGETS ${target} DESTINATION lib/${PROJECT_NAME})
endforeach()
ament_package()

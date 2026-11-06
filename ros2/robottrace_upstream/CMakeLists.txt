cmake_minimum_required(VERSION 3.16)
project(robottrace_upstream)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# 固定上游副本保持原样，仅在可再生构建目录应用补丁。
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/upstream")
file(COPY upstream/ DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/upstream")
execute_process(COMMAND patch --batch --fuzz=0 -p1 -i
    "${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.patch"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/upstream" RESULT_VARIABLE patched)
if(NOT patched EQUAL 0)
    message(FATAL_ERROR "Cannot apply RobotTrace upstream instrumentation")
endif()
set(upstream "${CMAKE_CURRENT_BINARY_DIR}/upstream")
include_directories(include ../robottrace_examples/include
    "${upstream}/demo_nodes_cpp/include" "${upstream}/topic_tools/include")
add_library(robottrace_demo_topics SHARED
    "${upstream}/demo_nodes_cpp/src/topics/talker.cpp"
    "${upstream}/demo_nodes_cpp/src/topics/listener.cpp")
target_compile_definitions(robottrace_demo_topics PRIVATE DEMO_NODES_CPP_BUILDING_DLL)
ament_target_dependencies(robottrace_demo_topics rclcpp rclcpp_components std_msgs)
rclcpp_components_register_node(robottrace_demo_topics
    PLUGIN "demo_nodes_cpp::Talker" EXECUTABLE talker)
rclcpp_components_register_node(robottrace_demo_topics
    PLUGIN "demo_nodes_cpp::Listener" EXECUTABLE listener)
add_library(robottrace_relay SHARED "${upstream}/topic_tools/src/relay_node.cpp"
    "${upstream}/topic_tools/src/tool_base_node.cpp")
target_compile_definitions(robottrace_relay PRIVATE TOPIC_TOOLS_BUILDING_LIBRARY)
ament_target_dependencies(robottrace_relay rclcpp rclcpp_components std_msgs)
add_executable(relay "${upstream}/topic_tools/src/relay.cpp")
ament_target_dependencies(relay rclcpp rclcpp_components)
target_link_libraries(relay robottrace_relay)
install(TARGETS robottrace_demo_topics robottrace_relay relay
    LIBRARY DESTINATION lib/${PROJECT_NAME} RUNTIME DESTINATION lib/${PROJECT_NAME})

enable_testing()
add_executable(identity_contract ../../tests/upstream_identity.cpp)
add_test(NAME identity_contract COMMAND identity_contract)
ament_package()

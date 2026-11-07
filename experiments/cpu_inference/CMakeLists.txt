cmake_minimum_required(VERSION 3.16)
project(robottrace_cpu_inference LANGUAGES CXX)

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.deps/onnxruntime-linux-x64-1.20.1"
    CACHE PATH "Matching official ONNX Runtime CPU headers and library")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ORT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ORT_LIBRARY onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(vision_inference vision.cpp)
target_compile_features(vision_inference PRIVATE cxx_std_17)
set_target_properties(vision_inference PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(vision_inference PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(vision_inference SYSTEM PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(vision_inference PRIVATE "${ORT_LIBRARY}")

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME vision_inference_reference
        COMMAND vision_inference "${CMAKE_CURRENT_SOURCE_DIR}/../../models/mobilenet")
    add_test(NAME vision_configuration_contract
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/test_characterization.py"
        "$<TARGET_FILE:vision_inference>" "${CMAKE_CURRENT_SOURCE_DIR}/../../models/mobilenet")
    set_tests_properties(vision_inference_reference PROPERTIES TIMEOUT 60)
    set_tests_properties(vision_configuration_contract PROPERTIES TIMEOUT 120)
endif()

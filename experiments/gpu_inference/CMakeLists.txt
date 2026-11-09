cmake_minimum_required(VERSION 3.16)
project(robottrace_gpu_verification LANGUAGES CXX)

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.deps/gpu/onnxruntime-linux-x64-gpu-1.20.1"
    CACHE PATH "Isolated ONNX Runtime 1.20.1 C/C++ GPU SDK")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ORT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ORT_LIBRARY onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(verify_cuda verify.cpp)
target_compile_features(verify_cuda PRIVATE cxx_std_17)
set_target_properties(verify_cuda PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(verify_cuda PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(verify_cuda SYSTEM PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(verify_cuda PRIVATE "${ORT_LIBRARY}")

add_executable(benchmark_inference benchmark.cpp)
target_compile_features(benchmark_inference PRIVATE cxx_std_17)
set_target_properties(benchmark_inference PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(benchmark_inference PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(benchmark_inference SYSTEM PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(benchmark_inference PRIVATE "${ORT_LIBRARY}")

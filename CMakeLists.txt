cmake_minimum_required(VERSION 3.24)
project(flashattention_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)

find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS /usr/include/x86_64-linux-gnu /usr/include)
find_library(CUDNN_LIBRARY cudnn PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR "cuDNN headers/library not found; required for library_composed_cublas_cudnn baseline")
endif()

option(FA_DEBUG_SYNC "Synchronize after kernel launches in debug builds" OFF)

add_library(flashattention_core
  kernels/flashattention_kernel.cu
  kernels/flashattention_v1.cu
  kernels/flashattention_v2.cu
  reference/flashattention_reference.cpp
  reference/flashattention_library_baseline.cpp
)
target_include_directories(flashattention_core PUBLIC include ${CUDNN_INCLUDE_DIR})
target_link_libraries(flashattention_core PUBLIC CUDA::cudart CUDA::cublas ${CUDNN_LIBRARY})
target_compile_options(flashattention_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)
if(FA_DEBUG_SYNC)
  target_compile_definitions(flashattention_core PUBLIC DEBUG_CUDA_SYNC=1)
endif()

add_executable(test_flashattention tests/test_flashattention.cpp)
target_link_libraries(test_flashattention PRIVATE flashattention_core)

add_executable(bench_flashattention benchmarks/bench_flashattention.cpp)
target_link_libraries(bench_flashattention PRIVATE flashattention_core)

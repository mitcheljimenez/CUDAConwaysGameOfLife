cmake_minimum_required(VERSION 3.22)
project(GameOfLife)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

add_subdirectory(extern)
add_subdirectory(test)

# add_library(matrix INTERFACE matrix.h)

add_executable(CudaGameOfLife main.cpp "cpu/SerialCPUGameOfLife.h" "cpu/main.cpp" "cuda-gpu/main.cpp" "cuda-gpu/GPUGameOfLife.cu")
# target_link_libraries(matrix_exec matrix)
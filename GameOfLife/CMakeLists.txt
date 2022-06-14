cmake_minimum_required(VERSION 3.22)

project(GameOfLife
	VERSION 1
	LANGUAGES CXX CUDA
)

enable_language(CUDA)

foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          14)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(GPUCudaGameOfLife main.cpp "cuda-gpu/GPUGameOfLife.h" "cuda-gpu/kernelWithIfs.cu")
add_executable(CPUGameOfLife cpu-test.cpp "cpu/SerialCPUGameOfLife.h")

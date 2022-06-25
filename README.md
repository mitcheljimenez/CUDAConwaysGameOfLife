# CUDAConwaysGameOfLife

Conway's parallel Game of life made on CUDA and OpenCL. There's a CPU serial version too just to
compare and benchmark.

### Compiling pre-requisites
- Having CUDA installed and a NVIDIA GPU (if running CUDA).

# Compilation
clone the repository into your computer:
```
git clone https://github.com/mitcheljimenez/CUDAConwaysGameOfLife.git
```
Once downloaded open CUDAConwaysGameOfLife folder with Visual Studio or with a terminal that suports cmake commands

1- Go to CUDAConwaysGameOfLife/GameOfLife folder in terminal (that supports mkdir and cmake commands preferably Visual Studio terminal)

2- Write the following code:
```
mkdir build
cd build
cmake ..
cmake --build .  //Or alternatevily Open GameOfLife/build/GameOfLife.sln with Visual Studio and build all solutions
```
This will throw some errors which can be ignored since they weren't implemented correctly.

For OpenCL Game of Life go to folder GameOfLife/opencl-gpu in terminal. If you are in folder CUDAConwaysGameOfLife/GameOfLife: 
```
cd opencl-gpu
```
Back in terminal make sure you are on GameOfLife/opencl-gpu and run the following code to test the openCL Game of Life with example file as initial board:
```
../build/opencl-gpu/Debug/OpenClGameOfLife inputs/input.dat inputs/input.params nx ny
```
Change the nx and ny for block size (nx * ny); they have to be 18%nx == 0 && 18%ny == 0 for the example.

This will print in console the initial board state and the final one and will create a program named final_state.dat in opengl-cpu folder which will have the final state of life.

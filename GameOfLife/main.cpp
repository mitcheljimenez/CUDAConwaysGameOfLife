using namespace std;
#include <stdio.h>
#include <iostream>
#include "cpu/SerialCPUGameOfLife.h"
#include "cuda-gpu/GPUGameOfLife.cu"

#define NUMBER_OF_ITERATIONS 1000

int main() {
	// We initialize cpu and gpu game of life 
	SerialCPUGameOfLife* cpuGameOfLife = new SerialCPUGameOfLife();
	GPUGameOfLife* gpuGameOfLife = new GPUGameOfLife();

	// Print first board state
	cpuGameOfLife->printBoard("CPU Board created:");
	gpuGameOfLife->printBoard("GPU Board created:");
	
	// Start game!
	cpuGameOfLife->startGame(NUMBER_OF_ITERATIONS);
	gpuGameOfLife->startGame(NUMBER_OF_ITERATIONS);

	// Print result board
	cpuGameOfLife->printBoard("CPU Board resulted:");
	gpuGameOfLife->printBoard("GPU Board resulted:");


}
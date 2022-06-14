using namespace std;
#include <stdio.h>
#include <iostream>
#include "cuda-gpu/GPUGameOfLife.h"

#define NUMBER_OF_ITERATIONS 1000

int main() {
	// We initialize cpu and gpu game of life 
	GPUGameOfLife* gpuGameOfLife = new GPUGameOfLife();

	// Print first board state
	gpuGameOfLife->printBoard("GPU Board created:");
	
	// Start game!
	gpuGameOfLife->startGame(NUMBER_OF_ITERATIONS);

	// Print result board
	gpuGameOfLife->printBoard("GPU Board resulted:");

}
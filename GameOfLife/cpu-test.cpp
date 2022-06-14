using namespace std;
#include <stdio.h>
#include <iostream>
#include "cpu/SerialCPUGameOfLife.h"

#define NUMBER_OF_ITERATIONS 1000

int main() {
	// We initialize cpu and gpu game of life 
	SerialCPUGameOfLife* cpuGameOfLife = new SerialCPUGameOfLife();

	// Print first board state
	cpuGameOfLife->printBoard("CPU Board created:");

	// Start game!
	cpuGameOfLife->startGame(NUMBER_OF_ITERATIONS);

	// Print result board
	cpuGameOfLife->printBoard("CPU Board resulted:");
}
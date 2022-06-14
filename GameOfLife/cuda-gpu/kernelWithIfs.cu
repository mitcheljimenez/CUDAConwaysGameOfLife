#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../countAliveCells.h"

#define BLOCK_SIZE 16
typedef unsigned char ubyte;

__global__ void kernelWithIfs(int boardWidth, int boardHeight, ubyte* currBoard, ubyte* nextBoard) {
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	int index = x * boardWidth + y;

	nextBoard[index] = currBoard[index];

	int y0 = ((y + boardHeight - 1) % boardHeight) * boardWidth;
	int y1 = y * boardWidth;
	int y2 = ((y + 1) % boardHeight) * boardWidth;

	int x0 = (x + boardWidth - 1) % boardWidth;
	int x2 = (x + 1) % boardWidth;

	ubyte neighbors = countAliveCells(currBoard, x0, x, x2, y0, y1, y2);

	if (neighbors < 2)
		nextBoard[index] = 0x0;

	if (neighbors > 3)
		nextBoard[index] = 0x0;

	if (neighbors == 3 && !currBoard[index])
		nextBoard[index] = 0x1;
}

void wrappedKernelWithIfs(int boardWidth, int boardHeight, ubyte* currBoard, ubyte* nextBoard) {
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize(boardWidth / BLOCK_SIZE, boardHeight / BLOCK_SIZE);
	
	kernelWithIfs< < < gridSize, blockSize > > >(boardWidth, boardHeight, currBoard, nextBoard);
}
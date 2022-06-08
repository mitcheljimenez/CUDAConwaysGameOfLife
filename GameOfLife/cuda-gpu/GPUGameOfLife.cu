#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

#define BLOCK_SIZE 16

typedef unsigned char ubyte;

class GPUGameOfLife {
private:
	int boardWidth;
	int boardHeight;
	ubyte* board;

public:
	GPUGameOfLife();
	GPUGameOfLife(int width, int height);
	~GPUGameOfLife();

	cudaError_t startGame(int numberOfIterations);
	void printBoard(string title);
};

GPUGameOfLife::GPUGameOfLife() : boardHeight(800), boardWidth(600) {
	board = (ubyte*)malloc(boardWidth * boardHeight * sizeof(ubyte));

	for (int i = 0; i < boardWidth; i++) {
		for (int j = 0; j < boardHeight; j++) {
			float random = rand() / (float)RAND_MAX;
			board[i * boardWith + j] = (random >= 0.5) ? 0x1 : 0x0;
		}
	}
}

GPUGameOfLife::GPUGameOfLife(int width, int height) : boardHeight(height), boardWidth(width) {
	board = (ubyte*)malloc(boardWidth * boardHeight * sizeof(ubyte));

	for (int i = 0; i < boardWidth; i++) {
		for (int j = 0; j < boardHeight; j++) {
			float random = rand() / (float)RAND_MAX;
			board[i * boardWith + j] = (random >= 0.5) ? 0x1 : 0x0;
		}
	}
}

GPUGameOfLife::~GPUGameOfLife() {
	delete board;
	delete boardHeight;
	delete boardWidth;
}


void GPUGameOfLife::printBoard(string title) {
	cout << "\n" << title << endl;

	for (int i = 0; i < boardWidth; i++) {
		for (int j = 0; j < boardHeight; j++) {
			cout << board[i * boardWidth + j] ? 'x' : ' ';
		}
		cout << "\n";
	}
}

static inline ubyte countAliveCells(ubyte* board, int x0, int x1, int x2, int y0, int y1, int y2) {
	return board[x0 + y0] + board[x1 + y0] + board[x2 + y0]
		+ board[x0 + y1] + board[x2 + y1]
		+ board[x0 + y2] + board[x1 + y2] + board[x2 + y2];
}


__global__ void kernelStepWithIfs(int rows, int cols, ubyte* cudaBoard, ubyte* cudaResultBoard) {
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	int index = x * cols + y;

	cudaResultBoard[index] = cudaBoard[index];

	int y0 = ((cols + boardHeight - 1) % boardHeight) * boardWidth;
	int y1 = cols * boardWidth;
	int y2 = ((cols + 1) % boardHeight) * boardWidth;

	int x0 = (rows + boardWidth - 1) % boardWidth;
	int x2 = (rows + 1) % boardWidth;

	int neighbors = countAliveCells(cudaBoard, x0, rows, x2, y0, y1, y2);

	if (neighbors < 2)
		cudaResultBoard[index] = 0x0;

	if (neighbors > 3)
		cudaResultBoard[index] = 0x0;

	if (neighbors == 3 && !cudaBoard[index])
		cudaResultBoard[index] = 0x1;
}

cudaError_t GPUGameOfLife::startGame(int numberOfIterations) {
	// Initialize needed variables
	cudaError_t cudaStatus;
	ubyte* cudaBoard;
	ubyte* cudaResultBoard;

	// Chose which GPU to run on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate all GPU buffers for boards
	cudaStatus = cudaMalloc((ubyte**)&cudaBoard, boardWidth * boardHeight * sizeof(ubyte));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((ubyte**)&cudaResultBoard, boardWidth * boardHeight * sizeof(ubyte));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy class board from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(cudaBoard, board, boardWidth * boardHeight * sizeof(ubyte), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Cuda mem set to cudaResultBoard
	cudaStatus = cudaMemset(cudaResultBoard, 0x0, boardWidth * boardHeight * sizeof(ubyte));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize(boardWidth / BLOCK_SIZE, boardHeight / BLOCK_SIZE);

	ubyte* currBoard;
	ubyte* nextBoard;

	for (int n = 0; n < numberOfIterations; n++) {
		if ((n % 2) == 0) {
			currBoard = cudaBoard;
			nextBoard = cudaResultBoard;
		} else {
			currBoard = cudaResultBoard;
			nextBoard = cudaBoard;
		}

		kernelStepWithIfs << <gridSize, blockSize >> > (boardWidth, boardHeight, currBoard, nextBoard);

		cudaMemcpy(board, cudaResultBoard, boardHeight * boardWidth * sizeof(ubyte), cudaMemcpyDeviceToHost);

		for (int i = 0; i < 24; i++) printf("\n");

		this->printBoard(" ");
	}

	cudaMemcpy(board, currBoard, boardWidth * boardHeight * sizeof(ubyte), cudaMemcpyDeviceToHost);

	this->printBoard("Resulting Board:");

	Error:
		cudaFree(cudaBoard);
		cudaFree(cudaResultBoard);
		return cudaStatus;
}
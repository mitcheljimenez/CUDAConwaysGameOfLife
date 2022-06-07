#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

#define BLOCK_SIDE 16

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

	void startGame(int numberOfIterations);
	void printBoard(string title);
};

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


__global__ void kernelStepWithIfs(int rows, int cols, ubyte* board, ubyte* resultBoard)
{
	int x = blockIdx.x * BLOCK_SIDE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIDE + threadIdx.y;

	int index = x * cols + y;

	resultBoard[index] = board[index];

	int y0 = ((cols + boardHeight - 1) % boardHeight) * boardWidth;
	int y1 = cols * boardWidth;
	int y2 = ((cols + 1) % boardHeight) * boardWidth;

	int x0 = (rows + m_worldWidth - 1) % m_worldWidth;
	int x2 = (rows + 1) % m_worldWidth;

	int neighbors = countAliveCells(board, x0, rows, x2, y0, y1, y2);

	if (neighbors < 2)
		resultBoard[index] = 0x0;

	if (neighbors > 3)
		resultBoard[index] = 0x0;

	if (neighbors == 3 && !board[index])
		resultBoard[index] = 0x1;
}
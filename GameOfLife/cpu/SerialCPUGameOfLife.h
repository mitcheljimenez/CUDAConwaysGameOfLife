#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

typedef unsigned char ubyte;

class SerialCPUGameOfLife {
	private:
		int boardWidth;
		int boardHeight;
		ubyte* board;

	public:
        SerialCPUGameOfLife();
        SerialCPUGameOfLife(int width, int height);
        ~SerialCPUGameOfLife();

        void startGame(int numberOfIterations);
        void printBoard(string title);
};

SerialCPUGameOfLife::SerialCPUGameOfLife() : boardHeight(800), boardWidth(600) {
    board = (ubyte*)malloc(boardWidth * boardHeight * sizeof(ubyte));

    for (int i = 0; i < boardWidth; i++) {
        for (int j = 0; j < boardHeight; j++) {
            float random = rand() / (float)RAND_MAX;
            board[i * boardWidth + j] = (random >= 0.5) ? 0x1 : 0x0;
        }
    }
}

SerialCPUGameOfLife::SerialCPUGameOfLife(int width, int height) : boardHeight(height), boardWidth(width) {
    board = (ubyte*)malloc(boardWidth * boardHeight * sizeof(ubyte));

    for (int i = 0; i < boardWidth; i++) {
        for (int j = 0; j < boardHeight; j++) {
            float random = rand() / (float)RAND_MAX;
            board[i * boardWidth + j] = (random >= 0.5) ? 0x1 : 0x0;
        }
    }
}

SerialCPUGameOfLife::~SerialCPUGameOfLife() {
    delete board;
    delete &boardHeight;
    delete &boardWidth;
}

void SerialCPUGameOfLife::printBoard(string title) {
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

void SerialCPUGameOfLife::startGame(int numberOfIterations) {
    ubyte* resultBoard = (ubyte*)malloc(boardWidth * boardHeight * sizeof(ubyte));

    for (int n = 0; n < numberOfIterations; n++) {
        for (int j = 0; j < boardHeight; j++) {
            int y0 = ((j + boardHeight - 1) % boardHeight) * boardWidth;
            int y1 = j * boardWidth;
            int y2 = ((j + 1) % boardHeight) * boardWidth;

            for (int i = 0; i < boardWidth; i++) {
                int x0 = (i + boardWidth - 1) % boardWidth;
                int x2 = (i + 1) % boardWidth;

                ubyte aliveCells = countAliveCells(board, x0, i, x2, y0, y1, y2);
                resultBoard[y1 + i] =
                    aliveCells == 3 || (aliveCells == 2 && board[i + y1]) ? 1 : 0;
            }
        }

        swap(board, resultBoard);
    }

    delete resultBoard;
}
typedef unsigned char ubyte;

static inline ubyte countAliveCells(ubyte* board, int x0, int x1, int x2, int y0, int y1, int y2) {
    return board[x0 + y0] + board[x1 + y0] + board[x2 + y0]
        + board[x0 + y1] + board[x2 + y1]
        + board[x0 + y2] + board[x1 + y2] + board[x2 + y2];
}
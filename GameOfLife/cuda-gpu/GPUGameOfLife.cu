#pragma once

template<typename T = int>
class GPUGameOfLife {
	private:
		size_t boardWidth = 600;
		size_t boardHeight = 800;

		ubyte* board;
		ubyte* resultBoard;
};
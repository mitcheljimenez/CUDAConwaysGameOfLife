using namespace std;
typedef unsigned char ubyte;

ubyte* m_data;
ubyte* m_resultData;

size_t m_worldWidth;
size_t m_worldHeight;
size_t m_dataLength;  // m_worldWidth * m_worldHeight

void computeIterationSerial() {
    for (size_t y = 0; y < m_worldHeight; ++y) {
        size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
        size_t y1 = y * m_worldWidth;
        size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

        for (size_t x = 0; x < m_worldWidth; ++x) {
            size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
            size_t x2 = (x + 1) % m_worldWidth;

            ubyte aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
            m_resultData[y1 + x] =
                aliveCells == 3 || (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
        }
    }

    swap(m_data, m_resultData);
}
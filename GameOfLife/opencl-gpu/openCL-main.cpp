#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "util.hpp"
#include "err_code.h"

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


#define FINALSTATEFILE "final_state.dat"

// Define the state of the cell
#define DEAD  0
#define ALIVE 1

/*************************************************************************************
 * Forward declarations of utility functions
 ************************************************************************************/
void die(const std::string message, const int line, const std::string file);
void load_board(std::vector<char>& board, const char* file, const unsigned int nx, const unsigned int ny);
void print_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny);
void save_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny);
void load_params(const char* file, unsigned int* nx, unsigned int* ny, unsigned int* iterations);
void load_chosen_params();
void load_random_board(const unsigned int nx, const unsigned int ny);
int load_chosen_blocksizeX();
int load_chosen_blocksizeY();





/*************************************************************************************
 * Main function
 ************************************************************************************/

int main(int argc, char** argv)
{

    // Check we have a starting state file
    if (argc != 5 && argc != 1)
    {
  
        printf("Usage:\n./No inputs or examples as described in readme\n");

        return EXIT_FAILURE;
    }
    else if(argc == 1) {
        load_chosen_params();
        argv[1] = "inputs/RandomInputs.dat";
        argv[2] = "inputs/ChosenParams.params";

        argv[3] = (char*)load_chosen_blocksizeX();
        argv[4] = (char*)load_chosen_blocksizeY();

        
    }
    // Board dimensions and iteration total
    unsigned int nx, ny;
    unsigned int bx = atoi(argv[3]);
    unsigned int by = atoi(argv[4]);
    unsigned int iterations;
    int platform_id = 0, device_id = 0;

    load_params(argv[2], &nx, &ny, &iterations);
    load_random_board(nx, ny);

    // Create OpenCL context, queue and program
    try
    {
        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        // Select the platform.
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

        // Create a context
        cl::Context context(devices);

        // Create a command queue
        // Select the device.
        cl::CommandQueue queue = cl::CommandQueue(context, devices[device_id]);
        cl::Program program(context, util::loadProgram("mykernel.cl"));
        try
        {
            program.build();
        }
        catch (cl::Error error)
        {
            // If it was a build error then show the error
            if (error.err() == CL_BUILD_PROGRAM_FAILURE)
            {
                std::vector<cl::Device> devices;
                devices = context.getInfo<CL_CONTEXT_DEVICES>();
                std::string built = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
                std::cerr << built << "\n";
            }
            throw error;
        }

        cl::make_kernel
            <cl::Buffer, cl::Buffer, unsigned int, unsigned int, cl::LocalSpaceArg>
            accelerate_life(program, "accelerate_life");

        // Allocate memory for boards
        std::vector<char> h_board(nx * ny);
        cl::Buffer d_board_tick(context, CL_MEM_READ_WRITE, sizeof(char) * nx * ny);
        cl::Buffer d_board_tock(context, CL_MEM_READ_WRITE, sizeof(char) * nx * ny);

        // Load in the starting state to host board and copy to device
        
        load_board(h_board, argv[1], nx, ny);
        queue.enqueueWriteBuffer(d_board_tick, CL_FALSE, 0, nx * ny * sizeof(char), &h_board.begin()); //Aca hay error

        // Display the starting state
        std::cout << "Starting state\n";
        print_board(h_board, nx, ny);

        // Set the global and local problem sizes
        cl::NDRange global(nx, ny);
        cl::NDRange local(bx, by);

        // Allocate local memory
        cl::LocalSpaceArg localmem = cl::Local(sizeof(char) * (bx + 2) * (by + 2));

        // Loop
        std::chrono::steady_clock::time_point ti = std::chrono::steady_clock::now();
        for (unsigned int i = 0; i < iterations; i++)
        {
            // Apply the rules of Life
            // Enqueue the kernel
            accelerate_life(cl::EnqueueArgs(queue, global, local), d_board_tick, d_board_tock, nx, ny, localmem);

            // Swap the boards over
            cl::Buffer tmp = d_board_tick;
            d_board_tick = d_board_tock;
            d_board_tock = tmp;
        }
        std::chrono::steady_clock::time_point tf = std::chrono::steady_clock::now();

        // Copy back the memory to the host
        queue.enqueueReadBuffer(d_board_tick, CL_FALSE, 0, nx * ny * sizeof(char), &h_board.begin());  //Aca hay error

        // Display the final state
        std::cout << "Finishing state\n";
        print_board(h_board, nx, ny);
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(tf - ti).count() << "[µs]" << std::endl;

        // Save the final state of the board
        save_board(h_board, nx, ny);

    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ":\n";
        err_code(err.err());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


/*************************************************************************************
 * Utility functions
 ************************************************************************************/

 // Function to load the params file and set up the X and Y dimensions
void load_params(const char* file, unsigned int* nx, unsigned int* ny, unsigned int* iterations)
{
    std::ifstream fp(file);
    if (!fp.is_open())
        die("Could not open params file.", __LINE__, __FILE__);

    int retval;
    fp >> *nx;
    fp >> *ny;
    fp >> *iterations;
    fp.close();
}

// Function to load in a file which lists the alive cells
// Each line of the file is expected to be: x y 1
void load_board(std::vector<char>& board, const char* file, const unsigned int nx, const unsigned int ny)
{
    std::ifstream fp(file);
    if (!fp.is_open())
        die("Could not open input file.", __LINE__, __FILE__);

    int retval;
    unsigned int x, y, s;
    while (fp >> x >> y >> s)
    {
        if (x > nx - 1)
            die("Input x-coord out of range.", __LINE__, __FILE__);
        if (y > ny - 1)
            die("Input y-coord out of range.", __LINE__, __FILE__);
        if (s != ALIVE)
            die("Alive value should be 1.", __LINE__, __FILE__);

        board[x + y * nx] = ALIVE;
    }

    fp.close();
}
//Function to choose the parameters of nx, ny and iterations
void load_chosen_params(){
    std::ofstream fp("inputs/ChosenParams.params");
    unsigned int x;
    unsigned int y;
    unsigned int ite;
    std::cout << "Choose Board Width" << std::endl;
    std::cin >> x;
    std::cout << "Choose Board Height" << std::endl;
    std::cin >> y;
    std::cout << "Choose Amount of Iterations" << std::endl;
    std::cin >> ite;
    fp << x << std::endl;
    fp << y << std::endl;
    fp << ite << std::endl;
    
}
//Functions to choose block sizes of nx, ny
int load_chosen_blocksizeX() {
    int x;
    std::cout << "Choose Block Width" << std::endl;
    std::cin >> x;
    return x;
}
int load_chosen_blocksizeY() {
    int y;
    std::cout << "Choose Block Height" << std::endl;
    std::cin >> y;
    return y;
}

// Function to load random values to each cell in board
void load_random_board(const unsigned int nx, const unsigned int ny)
{
    std::ofstream file("inputs/RandomInputs.dat");
    unsigned int total = nx * ny;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distrx(0, nx - 1);
    std::uniform_int_distribution<> distry(0, nx - 1);
    unsigned int randx = (unsigned int)distrx(gen);
    unsigned int randy = (unsigned int)distry(gen);
    for (int i = 0; i < randx; i++)
    {
        for (int j = 0; j < randy; j++)
        {
            file << distrx(gen) << " " << distry(gen) << 1 << std::endl;
        }
    }

}

// Function to print out the board to stdout
// Alive cells are displayed as O
// Dead cells are displayed as .
void print_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny)
{
    for (unsigned int i = 0; i < ny; i++)
    {
        for (unsigned int j = 0; j < nx; j++)
        {
            if (board[i * nx + j] == DEAD)
                std::cout << ".";
            else
                std::cout << "O";
        }
        std::cout << "\n";
    }
}

void save_board(const std::vector<char>& board, const unsigned int nx, const unsigned int ny)
{
    FILE* fp = fopen(FINALSTATEFILE, "w");
    if (!fp)
        die("Could not open final state file.", __LINE__, __FILE__);

    for (unsigned int i = 0; i < ny; i++)
    {
        for (unsigned int j = 0; j < nx; j++)
        {
            if (board[i * nx + j] == ALIVE)
                fprintf(fp, "%d %d %d\n", j, i, ALIVE);
        }
    }
}

// Function to display error and exit nicely
void die(const std::string message, const int line, const std::string file)
{
    std::cerr << "Error at line " << line << " of file " << file << ":\n";
    std::cerr << message << "\n";
    exit(EXIT_FAILURE);
}
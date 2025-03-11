#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <iostream>
#include <vector>
#include <random>

using namespace std;
using namespace alpaka;

/********************************************************************* */
// Step 1: Create the kernel
// Step 2: Decide where the parallel and non-parallel parts of the code run
// Step 3: Decide how to parallelize
// Step 4: Allocate memory on the host and device
// Step 5: Copy data to the device
// Step 6: Execute the kernel
// Step 7: Copy the result back to the host
/********************************************************************* */

// Define the kernel (Matrix-Matrix multiplication)
struct MyMxM {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float const* A, float const* B, float* C, size_t N) const {
        // Getting global thread indices 
        auto x = getIdx<Grid, Threads>(acc)[0];
        auto y = getIdx<Grid, Threads>(acc)[1];

        if (x < N && y < N) {
            float sum = 0.0f;
            // Matrix multiplication formula
            for (size_t k = 0; k < N; k++) {
                sum += A[y * N + k] * B[k * N + x];
            }
            // Store the result in C
            C[y * N + x] = sum;
        }
    }
};

// Initialize the matrix with some data
template<typename TM> 
inline void initializeMatrix(TM &span)
{
    auto const numColumns = span.extent(1);
    for(Idx i = 0; i < span.extent(0); ++i)
    {
        for(Idx j = 0; j < numColumns; ++j)
        {
            span(i, j) = static_cast<DataType>(i * numColumns + j);
        }
    }
}

auto main() -> int {
    using Dim = alpaka::DimInt<2>;  // 2D grid
    using Idx = std::size_t;        // Index type
    using DataType = std::uint32_t; // Data type for computation
    
    // Define the accelerator
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;  
    
    // Define the device associated with the accelerator
    using DevAcc = alpaka::Dev<Acc>;

    // Select a device from the platform of the accelerator
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);

    // Define matrix size and elements per thread
    Idx const N = 128;  // Example matrix size NxN
    Idx const elementsPerThread = 8u;

    // Allocate memory on the host and device 
    alpaka::Vec<Dim, Idx> const extent(N, N);  // 2D matrix
    using Data = std::uint32_t;
    
    // Allocate 3 host memory buffers (A, B, and C matrices)
    using BufHost = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufHost bufHostB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufHost bufHostC(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Random data generation for matrices A and B
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);
    
    for(Idx i(0); i < N; ++i)
    {
        for(Idx j(0); j < N; ++j)
        {
            bufHostA(i, j) = dist(eng);
            bufHostB(i, j) = dist(eng);
            bufHostC(i, j) = 0;
        }
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Create the queue for memcpy and kernel task
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue(devAcc);

    // Copy data from host to device
    alpaka::memcpy(queue, bufAccA, bufHostA);
    alpaka::memcpy(queue, bufAccB, bufHostB);

    // Define the kernel
    MyMxM kernel;

    // Let Alpaka calculate a good block and grid size based on the problem extent
    auto const workDiv = alpaka::workdiv::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        elementsPerThread);

    // Execute the kernel
    alpaka::exec<Acc>(
        queue,
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAccA),
        alpaka::mem::view::getPtrNative(bufAccB),
        alpaka::mem::view::getPtrNative(bufAccC),
        N); // Matrix size N

    // Copy the result back to the host
    alpaka::memcpy(queue, bufHostC, bufAccC);

    // Wait for the queue to finish
    alpaka::wait(queue);

    // Optionally: Print the result (first 5 elements for example)
    for (Idx i = 0; i < 5; ++i) {
        for (Idx j = 0; j < 5; ++j) {
            std::cout << bufHostC(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;           
}

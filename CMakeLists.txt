cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

# Manually add the Alpaka directory if it's not installed
set(ALPAKA_DIR "/home/mohammad/alpaka")
add_subdirectory(${ALPAKA_DIR} alpaka_build)

# Add the executable
add_executable(MyMxM MyMxM.cpp)

# Link Alpaka to your project
target_link_libraries(MyMxM alpaka)


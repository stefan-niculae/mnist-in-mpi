cmake_minimum_required(VERSION 3.5)
project(mnist_in_mpi)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp nn.cpp evaluate.cpp matrix.cpp reading.cpp utils.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mnist_in_mpi.out ${SOURCE_FILES})

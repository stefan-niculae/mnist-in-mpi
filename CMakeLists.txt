cmake_minimum_required(VERSION 3.5)
project(mnist_in_mpi)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mnist_in_mpi.out ${SOURCE_FILES})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
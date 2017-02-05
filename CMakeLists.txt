cmake_minimum_required(VERSION 3.6)
project(mnist)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(mnist ${SOURCE_FILES})
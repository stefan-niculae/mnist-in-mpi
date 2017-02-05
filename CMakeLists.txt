cmake_minimum_required(VERSION 3.6)
project(mnist)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mnist ${SOURCE_FILES})
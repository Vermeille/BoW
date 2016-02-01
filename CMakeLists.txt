cmake_minimum_required(VERSION 2.8)

project(NLP-COMMON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g3 -Wall -Wextra -Wno-deprecated-declarations")

add_subdirectory(autodiff/src)
add_subdirectory(src)
add_subdirectory(tests)



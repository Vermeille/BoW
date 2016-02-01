cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -g3 -Wno-deprecated-declarations")

add_subdirectory(httpi)
add_subdirectory(nlp-common)
add_subdirectory(src)


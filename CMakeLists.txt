cmake_minimum_required(VERSION 2.8.11)

project(BOW)

add_executable(bow bow.h bow.cpp main.cpp)
target_link_libraries(bow LINK_PUBLIC httpi glog gflags microhttpd)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

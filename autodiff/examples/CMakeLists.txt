cmake_minimum_required(VERSION 2.8)

project(libadd-examples CXX)

add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression ad)


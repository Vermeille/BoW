cmake_minimum_required(VERSION 2.8)

project(libad CXX)

find_package(Eigen3)

add_library(ad
    ad/graph.cpp
    ad/operators.cpp
    ad/ad.h
    ad/graph.h
    ad/operators.h
    ad/optimizer.h
    ad/optimizers/sgd.h
)

target_include_directories(ad PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})


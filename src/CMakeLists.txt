cmake_minimum_required(VERSION 2.8.11)

project(BOW)

add_executable(bow
    bow.h
    bow.cpp
    main.cpp
    pages/global.cpp
    pages/classify.cpp
    pages/pages.h
    pages/weights.cpp)

target_link_libraries(bow LINK_PUBLIC nlp-common httpi glog gflags microhttpd)

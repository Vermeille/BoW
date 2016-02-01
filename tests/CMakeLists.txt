cmake_minimum_required(VERSION 2.8)

project(nlp-test)

add_executable(test-rules rules-tests.cpp)
target_link_libraries(test-rules PUBLIC nlp-common)

add_executable(bow-classifier bow-classifier.cpp)
target_link_libraries(bow-classifier PUBLIC nlp-common)

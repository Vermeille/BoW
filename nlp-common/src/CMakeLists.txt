cmake_minimum_required(VERSION 2.8)

project(NLP-COMMON CXX)

add_library(nlp-common
    nlp/document.h
    nlp/rule.h
    nlp/rule.cpp
    nlp/rules-matcher.h
    nlp/rules-matcher.cpp
    nlp/tokenizer.h
    nlp/tokenizer.cpp
    nlp/featurizer.h
    nlp/featurizer.cpp
    nlp/dict.h
    nlp/dict.cpp
    nlp/bow.h
    nlp/bow.cpp
    nlp/sequence-tagger.cpp
    nlp/sequence-tagger.h
)

target_link_libraries(nlp-common LINK_PUBLIC glog ad)
target_include_directories(nlp-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


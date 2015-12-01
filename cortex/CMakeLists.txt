cmake_minimum_required(VERSION 3.3)
project(cortex)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lfftw3")

set(SOURCE_FILES
    constants.h
    libs.h
    main.cpp
    utils.cpp
    utils.h)

add_executable(cortex ${SOURCE_FILES})

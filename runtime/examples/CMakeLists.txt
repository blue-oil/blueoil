cmake_minimum_required(VERSION 3.5)

project(EXAMPLE)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_CXX_STANDARD "11")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/meta.yaml meta.yaml COPYONLY)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(a.out run.cpp)
target_link_libraries(a.out blueoil pthread)

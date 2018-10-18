cmake_minimum_required(VERSION 2.7 FATAL_ERROR)

project(packer CXX)
set(CMAKE_BUILD_TYPE Release)
if(DEFINED ENV{LM_CXX_C11})
	message("Use LM_CXX_C11")
	set(CMAKE_CXX_COMPILER $ENV{LM_CXX_C11})
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -Werror")
set(CMAKE_MACOSX_RPATH 1)

# Main source code (packer)
set(PACKER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
aux_source_directory(${PACKER_DIR}/src SRC_LIST)
include_directories(${PACKER_DIR}/header)

# Outputs
add_library(${PROJECT_NAME} MODULE ${SRC_LIST})


cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

# set(TOOLCHAIN_NAME "linux_arm")
set(TOOLCHAIN_NAME "linux-armv8a-hardfloat")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/toolchain/${TOOLCHAIN_NAME}.cmake")

project(progfpga CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS YES)

list(APPEND PROGFPGA_SRC
    main.cpp
)

include_directories(./thirdparty/)
include_directories(./include)
add_executable(progfpga ${PROGFPGA_SRC})


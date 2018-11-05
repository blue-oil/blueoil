cmake_minimum_required(VERSION 3.1)
project(conv_sample CXX)


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(ROOT_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)
get_filename_component(INTEL_HLS_DIR "${SRC_DIR}/intel_hls" ABSOLUTE)
get_filename_component(SRC_DIR ${ROOT_DIR}/src ABSOLUTE)
get_filename_component(TB_DIR "${SRC_DIR}/tb" ABSOLUTE)
get_filename_component(CPP_DIR "${SRC_DIR}/cpp" ABSOLUTE)

get_filename_component(INCLUDE_DIR "${ROOT_DIR}/include" ABSOLUTE)

if(WIN32)
    # for Windows operating system in general
elseif(MSVC OR MSYS OR MINGW)
    # for detecting Windows compilers
elseif(APPLE)
    # for MacOS X
elseif(UNIX AND NOT APPLE)
    # for Linux, BSD, Solaris, Minix
    # intel HLS
    get_filename_component(LOCAL_INTEL_HLS_INCLUDE_DIR "${ROOT_DIR}/include" ABSOLUTE)
    get_filename_component(GLOBAL_INTEL_HLS_INCLUDE_DIR "$ENV{HLS_INSTALL_DIR}/include" ABSOLUTE)
    get_filename_component(INTEL_HLS_LIB_DIR "$ENV{HLS_INSTALL_DIR}/host/linux64/lib/" ABSOLUTE)
endif()

include_directories(${INCLUDE_DIR})

file(GLOB_RECURSE TB_SRC "${TB_DIR}/*.cpp")
file(GLOB_RECURSE CPP_SRC "${CPP_DIR}/*.cpp")
file(GLOB_RECURSE INTEL_HLS_SRC "${INTEL_HLS_DIR}/*.cpp")
set(SRC ${TB_SRC} ${CPP_SRC})

enable_testing() 
add_executable(x86 ${SRC})
target_compile_options(x86 PRIVATE -std=c++0x
                                   -O0
                                   -Wall
                                   -Wextra
                                   -Werror
                                   -Wno-unknown-pragmas
                                   -Wno-unused-function
                                   -Wno-unused-parameter
                                   -Wno-unused-variable
                                   -Wno-sign-compare
                                   -fno-omit-frame-pointer
                                   -fsanitize=address)
target_link_libraries(x86
  -fsanitize=address
)
add_test(NAME x86_test 
         COMMAND x86 random
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})


if(UNIX AND NOT APPLE)
    set(HLS_COMPONENT --component intel_hls_qconv3x3_impl)

    add_executable(sim ${SRC} ${INTEL_HLS_SRC} include/cpp_utils.h include/cpp/gemm.cpp)
    target_compile_options(sim PRIVATE -march=x86-64 ${HLS_COMPONENT})
    target_compile_definitions(sim PRIVATE _INTEL_HLS_)
    target_link_libraries(sim PRIVATE m)
    target_include_directories(sim PRIVATE ${HLS_INCLUDE_DIR})

    add_executable(syn ${SRC} ${INTEL_HLS_SRC})
    target_compile_options(syn PRIVATE -march=5CSEBA6U23I7 ${HLS_COMPONENT} -ghdl --clock 50MHz)
    target_compile_definitions(syn PRIVATE _INTEL_HLS_)
    # get_filename_component(INTEL_HLS_LIB_DIR "${INTEL_HLS_LIB_DIR}/" ABSOLUTE)
    target_include_directories(syn PRIVATE ${HLS_INCLUDE_DIR})
endif()

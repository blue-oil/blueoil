cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

#
# Build configuration from config file
#
set(CONFIG_FILE ${CMAKE_SOURCE_DIR}/config.cmake)
if(EXISTS ${CONFIG_FILE})
    include(${CONFIG_FILE})
endif()

#
# Build configuration from command line
#
if(TOOLCHAIN_NAME)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/toolchain/${TOOLCHAIN_NAME}.cmake")
endif()


#
# The project
#
project(DlkRunTime CXX ASM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS YES)

#
# Common code
#
file(GLOB SRC_LIB_ALL "src/inputs/*.cpp")
list(APPEND SRC_LIB_ALL
    src/func/add.cpp
    src/func/average_pool.cpp
    src/func/bias_add.cpp
    src/func/conv2d.cpp
    src/func/impl/apply_thresholds.cpp
    src/func/impl/quantized_conv2d_dim2col.cpp
    src/func/max.cpp
    src/func/max_pool.cpp
    src/func/minimum.cpp
    src/func/pad.cpp
    src/func/mul.cpp
    src/func/matmul.cpp
    src/func/quantize.cpp
    src/func/quantized_conv2d.cpp
    src/func/real_div.cpp
    src/func/relu.cpp
    src/func/leaky_relu.cpp
    src/func/round.cpp
    src/func/scale.cpp
    src/func/softmax.cpp
    src/func/sqrt.cpp
    src/func/sub.cpp
    src/func/unpooling.cpp
    src/matrix/shift_add.cpp
    src/network_c_interface.cpp
    src/network.cpp
    src/pack_input_to_qwords.cpp
    src/time_measurement.cpp
)

if(EXISTS ${CMAKE_SOURCE_DIR}/src/scaling_factors.cpp)
    list(APPEND SRC_LIB_ALL src/scaling_factors.cpp)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/src/thresholds.cpp)
    list(APPEND SRC_LIB_ALL src/thresholds.cpp)
endif()

if(RUN_ON_FPGA)
    list(APPEND SRC_LIB_ALL src/pack2b_neonv7.S)
    list(APPEND SRC_LIB_ALL src/func/arm_neon/batch_normalization.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/arm_neon/quantized_conv2d_tiling.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/fpga/quantized_conv2d_kn2row.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/arm_neon/pop_count.cpp)
    list(APPEND SRC_LIB_ALL src/matrix/arm_neon/quantized_multiplication.cpp)
elseif(USE_NEON)
    list(APPEND SRC_LIB_ALL src/pack2b_neonv7.S)
    list(APPEND SRC_LIB_ALL src/func/arm_neon/batch_normalization.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/arm_neon/quantized_conv2d_tiling.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/generic/quantized_conv2d_kn2row.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/arm_neon/pop_count.cpp)
    list(APPEND SRC_LIB_ALL src/matrix/arm_neon/quantized_multiplication.cpp)
else()
    list(APPEND SRC_LIB_ALL src/func/generic/batch_normalization.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/generic/quantized_conv2d_kn2row.cpp)
    list(APPEND SRC_LIB_ALL src/func/impl/generic/pop_count.cpp)
    list(APPEND SRC_LIB_ALL src/matrix/generic/quantized_multiplication.cpp)
endif()

#
# Debug binary code
#
file(GLOB SRC_DEBUG_BINARY "mains/main.cpp")

#
# Compilation & Linking flags / Includes
#
macro(add_dlk_target_compile_properties target)
    target_compile_options(${target} PUBLIC -pthread)
    target_include_directories(${target} PUBLIC include)
    if(USE_NEON)
        target_compile_definitions(${target} PUBLIC -DUSE_NEON)
        target_compile_options(${target} PUBLIC -mcpu=cortex-a9 -mfpu=neon -mthumb -fopenmp)
        target_link_libraries(${target} PUBLIC -fopenmp)
    endif()
    if(RUN_ON_FPGA)
        target_compile_definitions(${target} PUBLIC -DRUN_ON_FPGA)
    endif()
endmacro()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

if(RUN_ON_FPGA)
    set(FPGA_SUFFIX "_fpga")
endif()

#
# Debug binaries
#
add_executable(lm ${SRC_DEBUG_BINARY} ${SRC_LIB_ALL})
set_target_properties(lm PROPERTIES SUFFIX "_${CMAKE_SYSTEM_PROCESSOR}${FPGA_SUFFIX}")
add_dlk_target_compile_properties(lm)
target_compile_definitions(lm PUBLIC -DFUNC_TIME_MEASUREMENT)

#
# Shared library
#
add_library(lib SHARED ${SRC_LIB_ALL})
add_dlk_target_compile_properties(lib)
set_target_properties(lib PROPERTIES PREFIX "")
set_target_properties(lib PROPERTIES SUFFIX "_${CMAKE_SYSTEM_PROCESSOR}${FPGA_SUFFIX}.so")
set_target_properties(lib PROPERTIES CXX_VISIBILITY_PRESET hidden)

#
# Tests and benchmarks
#
enable_testing()
add_subdirectory(test)


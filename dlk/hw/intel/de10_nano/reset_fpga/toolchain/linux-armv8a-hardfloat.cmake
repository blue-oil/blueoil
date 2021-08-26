#
# Please, if you want to use a specific toolchain and sysroot
# define TOOLCHAIN_PATH and CMAKE_SYSROOT variables:
#
set(TOOLCHAIN_PATH "")
set(CMAKE_SYSROOT "")
set(AARCH32 ON)

#
# Cross-compiling for ARM
#
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(NOT TOOLCHAIN_PATH)
    set(TOOLCHAIN_BIN_PATH "")
else()
    set(TOOLCHAIN_BIN_PATH ${TOOLCHAIN_PATH}/bin/)
endif()

set(CMAKE_CXX_COMPILER ${TOOLCHAIN_BIN_PATH}arm-linux-gnueabihf-g++-8)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_BIN_PATH}arm-linux-gnueabihf-g++-8)
set(CMAKE_CXX_COMPILER_AR ${TOOLCHAIN_BIN_PATH}arm-linux-gnueabihf-ar)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

cmake_minimum_required(VERSION 3.5)

project(DCORE CXX)

set(DCORE_VERSION_MAJOR "0")
set(DCORE_VERSION_MINOR "1")
set(DCORE_VERSION_PATCH "0")
set(DCORE_VERSION "${DCORE_VERSION_MAJOR}.${DCORE_VERSION_MINOR}.${DCORE_VERSION_PATCH}")


option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)

if(BUILD_SHARED_LIBS)
  # When BUILD_SHARED_LIBS is ON, override BUILD_SHARED_LIBS is OFF to build static yaml-cpp library.
  set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "Set BUILD_SHARED_LIBS OFF for yaml cpp" FORCE)
  set(DCORE_BUILD_SHARED_LIBS ON)

else()

  set(DCORE_BUILD_SHARED_LIBS OFF)
endif()

option(USE_OPENCV "use OpenCV library" OFF)
if(USE_OPENCV)
  find_package(OpenCV REQUIRED)
  if(NOT OpenCV_FOUND)
    message(SEND_ERROR "OpenCV library not found.")
  endif()
  add_definitions(-DUSE_OPENCV=1)
endif()

option(USE_LIBPNG "use PNG library" OFF)
if(NOT USE_OPENCV)
  if(USE_LIBPNG)
    find_package(PNG REQUIRED)
    if(NOT PNG_FOUND)
      message(SEND_ERROR "PNG library not found.")
    endif()
    add_definitions(-DUSE_LIBPNG=1)
  endif()
endif()

# Add yaml cpp
option(YAML_CPP_BUILD_TESTS "Enable testing" OFF)
option(YAML_CPP_BUILD_TOOLS "Enable parse tools" OFF)
option(YAML_CPP_BUILD_CONTRIB "Enable contrib stuff in library" OFF)
add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)

# set cxx 
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-Wall -O2")
set(CMAKE_CXX_STANDARD 11)

# output directory of `make install`
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/output) # dir for output


file(GLOB SRCS "src/[a-zA-Z]*.cpp")
file(GLOB PUBLIC_HEADERS "include/blueoil.hpp")
file(GLOB PRIVATE_HEADERS "include/[a-zA-Z]*.hpp")
if(NOT USE_OPENCV)
  file(GLOB OPENCV_SRCS "src/*_opencv.cpp")
  file(GLOB OPENCV_HEADERS "include/*_opencv.hpp")
  list(REMOVE_ITEM SRCS ${OPENCV_SRCS})
  list(REMOVE_ITEM PRIVATE_HEADERS ${OPENCV_HEADERS})
endif()
if(NOT USE_LIBPNG)
  file(GLOB PNG_SRCS "src/*_png.cpp")
  file(GLOB PNG_HEADERS "include/*_png.hpp")
  list(REMOVE_ITEM SRCS ${PNG_SRCS})
  list(REMOVE_ITEM PRIVATE_HEADERS ${PNG_HEADERS})
endif()


if(VERBOSE)
  message(STATUS "use_opencv: ${USE_OPENCV}")
  message(STATUS "use_libpng: ${USE_LIBPNG}")
  message(STATUS "sources: ${SRCS}")
  message(STATUS "public_headers: ${PUBLIC_HEADERS}")
  message(STATUS "private_headers: ${PRIVATE_HEADERS}")
endif()


# add library headers
include_directories(include yaml-cpp/include)
if(USE_OPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()
if(USE_PNG)
  include_directories(${PNG_INCLUDE_DIRS})
endif()


# Add dlk static lib.
add_library(dlk STATIC IMPORTED)
if(DEFINED ENV{DLK_LIB_DIR})
  set(DLK_LIB_DIR $ENV{DLK_LIB_DIR})
else()
  set(DLK_LIB_DIR ${CMAKE_SOURCE_DIR}/dlk_lib)
endif()
message(STATUS "dlk dir: ${DLK_LIB_DIR}")

# TODO(wakisaka): Enable to set path, target environment.
set_target_properties(dlk PROPERTIES IMPORTED_LOCATION ${DLK_LIB_DIR}/libdlk_x86.a)


if(DCORE_BUILD_SHARED_LIBS)

  # Build static yaml-cpp library with `-fPIC` flags.
  get_target_property(YAML_CPP_COMPILE_FLAGS yaml-cpp COMPILE_FLAGS)
  set_target_properties(yaml-cpp PROPERTIES COMPILE_FLAGS "${YAML_CPP_COMPILE_FLAGS} -fPIC")

  set(DCORE_LIB ${CMAKE_BINARY_DIR}/libblueoil.so)
  add_library(blueoil SHARED ${SRCS} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
  set_target_properties(blueoil PROPERTIES
	VERSION "${DCORE_VERSION}"
	SOVERSION "${DCORE_VERSION_MAJOR}.${DCORE_VERSION_MINOR}.${DCORE_VERSION_PATCH}"
    PUBLIC_HEADER ${PUBLIC_HEADERS}
    )
  target_link_libraries(blueoil dlk yaml-cpp)

  install(TARGETS blueoil
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
    )

else()
  add_library(tmp STATIC ${SRCS} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})

  set(DCORE_LIB ${CMAKE_BINARY_DIR}/libblueoil.a)
  set(COMBINED_OBJECT _combined_object) # dir storage of objects.
  add_custom_target(blueoil ALL
    COMMAND rm -rf ${COMBINED_OBJECT}
    COMMAND mkdir ${COMBINED_OBJECT}
    COMMAND cd ${COMBINED_OBJECT} && ar -x $<TARGET_FILE:yaml-cpp>
    COMMAND cd ${COMBINED_OBJECT} && ar -x $<TARGET_FILE:tmp>
    COMMAND cd ${COMBINED_OBJECT} && ar -x $<TARGET_FILE:dlk>
    COMMAND ar -qcs ${DCORE_LIB} ${COMBINED_OBJECT}/*.o
    COMMAND rm -rf ${COMBINED_OBJECT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS yaml-cpp tmp dlk
    COMMENT "Create ${DCORE_LIB} with combining static libs yaml-cpp, tmp. dlk."
    )


  install(FILES ${DCORE_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  install(FILES ${PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

endif()

# CTest
if(DCORE_BUILD_SHARED_LIBS)
  enable_testing()
  add_subdirectory(test)
endif()

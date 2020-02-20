set(CMAKE_THREAD_LIBS_INIT "-lpthread")

include(ExternalProject)

# Include gtest as a dependency
list(APPEND CMAKE_ARGS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
list(APPEND CMAKE_ARGS "-DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON")
list(APPEND CMAKE_ARGS "-DCMAKE_BUILD_TYPE=RELEASE")

# list(APPEND CMAKE_ARGS "-DRUN_HAVE_STD_REGEX=0")
# list(APPEND CMAKE_ARGS "-DRUN_HAVE_GNU_POSIX_REGEX=0")
list(APPEND CMAKE_ARGS "-DRUN_HAVE_POSIX_REGEX=0")
# list(APPEND CMAKE_ARGS "-DRUN_HAVE_STEADY_CLOCK=0")

# Download and install GoogleTest
ExternalProject_Add(
    gbench
    URL https://github.com/google/benchmark/archive/master.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gbench
    INSTALL_COMMAND ""
    CMAKE_ARGS ${CMAKE_ARGS})

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gbench source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gbench)

set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googletest/lib/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gbench)

set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googletest/lib/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libbenchmark target to be used as a dependency by test programs
add_library(libbenchmark IMPORTED STATIC GLOBAL)
add_dependencies(libbenchmark gbench)

set_target_properties(libbenchmark PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/src/libbenchmark.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

include_directories("${binary_dir}/googletest/include"
                    "${source_dir}/include")

add_subdirectory(testBuffer)

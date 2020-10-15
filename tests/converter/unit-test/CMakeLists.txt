cmake_minimum_required(VERSION 2.8.11)
project(converter_unit_test
    LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(third/DownloadProject/DownloadProject.cmake)
download_project(PROJ                googletest
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             release-1.10.0
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(EXECUTE_FILE "converter_unit_test")

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()

set(CONVERTER
    "../../../../blueoil/converter/templates/src/time_measurement.cpp"
    "../../../../blueoil/converter/templates/src/func/impl/generic/pop_count.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE EXECUTE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.cpp")
add_executable(${EXECUTE_FILE} ${EXECUTE} ${CONVERTER})
target_compile_features(${EXECUTE_FILE} PRIVATE cxx_std_14)

target_link_libraries(${EXECUTE_FILE} gtest gmock_main)
add_test(NAME unittest COMMAND ${EXECUTE_FILE})

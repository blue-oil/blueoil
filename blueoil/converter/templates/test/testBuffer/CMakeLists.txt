file(GLOB SRC *.cpp)

add_executable(testBuffer ${SRC})

target_link_libraries(
    testBuffer
    libgtest
    libgmock
    libbenchmark
)

add_test(testBuffer testBuffer)

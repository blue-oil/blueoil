add_executable(blueoil-test-tensor test_tensor.cpp)
add_executable(blueoil-test-image  test_image.cpp)
add_executable(blueoil-test-opencv test_opencv.cpp)
add_executable(blueoil-test-resize test_resize.cpp)

find_package( OpenCV REQUIRED )
target_link_libraries(blueoil-test-tensor blueoil ${OpenCV_LIBS})
target_link_libraries(blueoil-test-image  blueoil ${OpenCV_LIBS})
target_link_libraries(blueoil-test-opencv blueoil ${OpenCV_LIBS})
target_link_libraries(blueoil-test-resize blueoil ${OpenCV_LIBS})

add_test(blueoil-test-tensor blueoil-test-tensor)
add_test(blueoil-test-image  blueoil-test-image)
add_test(blueoil-test-opencv blueoil-test-opencv)
add_test(blueoil-test-resize blueoil-test-resize)

# test images for opencv
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/images
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

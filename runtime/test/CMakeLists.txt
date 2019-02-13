find_package( OpenCV REQUIRED )

macro(blueoil_unittest item)
  add_executable(blueoil-test-${item} test_${item}.cpp test_util.cpp)
  target_link_libraries(blueoil-test-${item} blueoil ${OpenCV_LIBS})
  add_test(blueoil-test-${item} blueoil-test-${item})
endmacro(blueoil_unittest)

blueoil_unittest(tensor)
blueoil_unittest(image)
blueoil_unittest(opencv)
blueoil_unittest(resize)

# test images for opencv
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/images
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(USE_OPENCV)
  find_package( OpenCV REQUIRED )
  if(NOT OpenCV_FOUND)
    message(SEND_ERROR "OpenCV library not found.")
  endif()
endif()

macro(blueoil_unittest item)
  add_executable(blueoil-test-${item} test_${item}.cpp test_util.cpp)
  if(USE_OPENCV)
    target_link_libraries(blueoil-test-${item} blueoil ${OpenCV_LIBS})
  else()
    target_link_libraries(blueoil-test-${item} blueoil "-pthread")
  endif()
  add_test(blueoil-test-${item} blueoil-test-${item})
endmacro(blueoil_unittest)

blueoil_unittest(tensor)
blueoil_unittest(image)
if(USE_OPENCV)
  blueoil_unittest(opencv)
endif()
blueoil_unittest(resize)
blueoil_unittest(data_processor)

# test images for opencv
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/images
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

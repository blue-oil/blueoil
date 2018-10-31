#include <cstdlib>
#include <iostream>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_opencv.hpp"

float test_expect[3][3][3] =
    { { // Red
       {255,255,255},
       {255,170,  0},
       {  0,  0,  0},
       },
      { // Green
       {  0,255,255},
       {  0,170,255},
       {  0,  0,255},
      },
      { // Blue
       {  0,  0,255},
       {255,170,  0},
       {  0,255,255},
      } };

int test_opencv() {
    cv::Mat img = cv::imread("images/3x3colors.png"); // PNG24 using 9 colors
    blueoil::Tensor input = blueoil::opencv::Tensor_fromCVMat(img);
    blueoil::Tensor expect = blueoil::Tensor::array({3, 3, 3} ,
						    (float *)test_expect);
    expect = blueoil::image::Tensor_CHW_to_HWC(expect);
    if (! input.allequal(expect)) {
	std::cerr << "test_opencv: input != expect" << std::endl;
	input = blueoil::image::Tensor_HWC_to_CHW(input);
	expect = blueoil::image::Tensor_HWC_to_CHW(expect);
	input.dump();
	expect.dump();
	return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main(void) {
    int status_code = 0;
    status_code = test_opencv();
    if (status_code != EXIT_FAILURE) {
	std::exit(status_code);
    }
    std::exit(EXIT_SUCCESS);
}

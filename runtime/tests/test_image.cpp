#include <cstdlib>
#include <iostream>
#include "blueoil.hpp"
#include "blueoil_image.hpp"

float test_chw_data[3][2][2] =  // planar RGB format
    { {  // Red
       {255,255},
       {  0,  0},
       },
      {  // Green
       {  0,255},
       {  0,255},
      },
      {  // Blue
       {  0,  0},
       {255,  0},
      } };


float test_hwc_data[2][2][3] =  // packed RGB format
    {
     { {255,  0,  0}, {255,255,  0} },  // R Y
     { {  0,  0,255}, {  0,255,  0} },  // B G
    };

int test_image() {
    blueoil::Tensor test_chw = blueoil::Tensor::array({3, 2, 2},
						      (float *) test_chw_data);
    blueoil::Tensor test_hwc = blueoil::Tensor::array({2, 2, 3},
						      (float *) test_hwc_data);
    blueoil::Tensor test_chw_hwc = blueoil::image::Tensor_CHW_to_HWC(test_chw);
    blueoil::Tensor test_hwc_chw = blueoil::image::Tensor_HWC_to_CHW(test_hwc);

    float *rgb = blueoil::image::Tensor_at(test_hwc, 1, 0);
    if ((rgb[0] != 255) || (rgb[1] != 255) || (rgb[2] != 0)) {
	std::cerr << "test_image: at(1, 0) != (255,255,0)" << std::endl;
	test_hwc.dump();
	return EXIT_FAILURE;
    }
    if (test_chw.allequal(test_hwc_chw) == false) {
	std::cerr << "test_image: test_chw != test_hwc_chw" << std::endl;
	test_chw.dump();
	test_hwc_chw.dump();
	return EXIT_FAILURE;
    }
    if (test_hwc.allequal(test_chw_hwc) == false) {
	std::cerr << "test_image: test_hwc != test_chw_hwc" << std::endl;
	test_hwc.dump();
	test_chw_hwc.dump();
	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(void) {
    int status_code = test_image();
    std::exit(status_code);
}


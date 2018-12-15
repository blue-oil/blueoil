#include <cstdlib>
#include <iostream>
#include "blueoil.hpp"
#include "tensor_util.hpp"

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
    blueoil::Tensor test_chw({3, 2, 2}, (float *) test_chw_data);
    blueoil::Tensor test_hwc({2, 2, 3}, (float *) test_hwc_data);
    blueoil::Tensor test_chw_hwc = blueoil::util::Tensor_CHW_to_HWC(test_chw);
    blueoil::Tensor test_hwc_chw = blueoil::util::Tensor_HWC_to_CHW(test_hwc);

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


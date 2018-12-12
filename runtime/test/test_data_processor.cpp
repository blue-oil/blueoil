#include <cstdlib>
#include <iostream>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_opencv.hpp"
#include "blueoil_data_processor.hpp"

float test_input[3][8][8] =
    { { // Red
       {255, 0, 0, 0, 0, 0, 0, 0},
       {0, 255, 0, 0, 0, 0, 0, 0},
       {0, 0, 100, 0, 0, 0, 0, 0},
       {0, 0, 0, 100, 0, 0, 0, 0},
       {0, 0, 0, 0, 100, 0, 0, 0},
       {0, 0, 0, 0, 0, 100, 0, 0},
       {0, 0, 0, 0, 0, 0, 255, 0},
       {0, 0, 0, 0, 0, 0, 0, 255}
       },
      { // Green
       {0, 0, 0, 0, 0, 0, 0, 255},
       {0, 0, 0, 0, 0, 0, 255, 0},
       {0, 0, 0, 0, 0, 100, 0, 0},
       {0, 0, 0, 0, 100, 0, 0, 0},
       {0, 0, 0, 100, 0, 0, 0, 0},
       {0, 0, 100, 0, 0, 0, 0, 0},
       {0, 255, 0, 0, 0, 0, 0, 0},
       {255, 0, 0, 0, 0, 0, 0, 0}
      },
      { // Blue
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0},
       {255,255,255,100,100,255,255,255},
       {255,255,255,100,100,255,255,255},
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0}
      } };

float test_expect[3][4][4] =
    { { // Red
       {255,  0, 0,   0},
       {  0,100, 0,   0},
       {  0,  0,100,  0},
       {  0,  0,  0,255}
       },
      { // Green
       {  0,  0,  0,  0},
       {  0,  0,  0,  0},
       {  0,  0,  0,  0},
       {  0,  0,  0,  0}
      },
      { // Blue
       {  0,  0,255,  0},
       {  0,  0,255,  0},
       {255,255,100,255},
       {  0,  0,255,  0}
      } };

int test_data_processor_resize() {
    // CHW (3-channel, height, width)
    int width = 4, height = 4;
    const std::pair<int, int>& image_size = std::make_pair(width, height);
    blueoil::Tensor input({3, 8, 8}, (float *)test_input);
    blueoil::Tensor expect({3, 4, 4}, (float *)test_expect);
    input = blueoil::image::Tensor_CHW_to_HWC(input);
    expect = blueoil::image::Tensor_CHW_to_HWC(expect);
    blueoil::Tensor output = blueoil::data_processor::Resize(input,
							     image_size);
    if (! output.allclose(expect)) {
	std::cerr << "test_data_processor_resize: output != expect" << std::endl;
	output = blueoil::image::Tensor_HWC_to_CHW(output);
	expect = blueoil::image::Tensor_HWC_to_CHW(expect);
	output.dump();
	expect.dump();
	return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main(void) {
    int status_code = 0;
    status_code = test_data_processor_resize();
    if (status_code != EXIT_FAILURE) {
	std::exit(status_code);
    }
    std::exit(EXIT_SUCCESS);
}

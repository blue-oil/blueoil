#include <cstdlib>
#include <iostream>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_opencv.hpp"
#include "test_util.hpp"

float test_expect[3][8][8] =
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

int test_load() {
  // CHW (3-channel, height, width)
  blueoil::Tensor input = blueoil::Tensor_loadImage("images/testinput.png");
  blueoil::Tensor expect({3, 8, 8}, (float *)test_expect);
  expect = blueoil::util::Tensor_CHW_to_HWC(expect);
  if (! input.allclose(expect)) {
    std::cerr << "test_resize: input != expect" << std::endl;
    blueoil::util::Tensor_HWC_to_CHW(input).dump();
    blueoil::util::Tensor_HWC_to_CHW(expect).dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(void) {
  int status_code = 0;
  status_code = test_load();
  if (status_code != EXIT_FAILURE) {
    std::exit(status_code);
  }
  std::exit(EXIT_SUCCESS);
}

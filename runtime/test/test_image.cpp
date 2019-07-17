/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

#include <cstdlib>
#include <iostream>
#include "blueoil.hpp"
#include "test_util.hpp"

float test_chw_data[3][2][2] =  // planar RGB format
  { {  // Red
     {255, 255},
     {  0,   0},
     },
    {  // Green
     {  0, 255},
     {  0, 255},
    },
    {  // Blue
     {  0,  0},
     {255,  0},
    } };


float test_hwc_data[2][2][3] = {  // packed RGB format
  { {255,  0,   0}, {255, 255,  0} },  // R Y
  { {  0,  0, 255}, {  0, 255,  0} },  // B G
};

int test_image() {
  blueoil::Tensor test_chw({3, 2, 2}, reinterpret_cast<float *>(test_chw_data));
  blueoil::Tensor test_hwc({2, 2, 3}, reinterpret_cast<float *>(test_hwc_data));
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


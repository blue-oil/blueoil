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
#include "blueoil_npy.hpp"
#include "test_util.hpp"

float test_expect[3][3][3] =
  { {  // Red
     {255, 255, 255},
     {255, 170,   0},
     {  0,   0,   0},
     },
    {  // Green
     {  0, 255, 255},
     {  0, 170, 255},
     {  0,   0, 255},
    },
    {  // Blue
     {  0,   0, 255},
     {255, 170,   0},
     {  0, 255, 255},
    } };

int test_npy() {
  blueoil::Tensor input = blueoil::npy::Tensor_fromNPYFile("images/3x3colors.npy");
  blueoil::Tensor input_float = blueoil::npy::Tensor_fromNPYFile("images/3x3colors-float.npy");
  blueoil::Tensor expect({3, 3, 3}, reinterpret_cast<float *>(test_expect));
  expect = blueoil::util::Tensor_CHW_to_HWC(expect);
  if (!input.allequal(expect)) {
    std::cerr << "test_npy: input != expect" << std::endl;
    blueoil::util::Tensor_HWC_to_CHW(input).dump();
    blueoil::util::Tensor_HWC_to_CHW(expect).dump();
    return EXIT_FAILURE;
  }
  if (!input_float.allequal(expect)) {
    std::cerr << "test_npy: input_float != expect" << std::endl;
    blueoil::util::Tensor_HWC_to_CHW(input_float).dump();
    blueoil::util::Tensor_HWC_to_CHW(expect).dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(void) {
  int status_code = 0;
  status_code = test_npy();
  if (status_code != EXIT_FAILURE) {
    std::exit(status_code);
  }
  std::exit(EXIT_SUCCESS);
}

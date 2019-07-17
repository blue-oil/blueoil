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

int test_tensor() {
  float tensor_data[][3] = {
                            {1, 2, 3},
                            {7, 8, 9}
  };
  float tensor_data2[][2] = {  // equals data, different shape.
                             {1, 2},
                             {3, 7},
                             {8, 9}
  };
  float tensor_data3[][3] = {  // equals shape, different data.
                             {1, 2, 3},
                             {7, 0, 9}
  };
  blueoil::Tensor tensor0({2, 3}, reinterpret_cast<float*>(tensor_data));
  blueoil::Tensor tensor1({2, 3}, reinterpret_cast<float*>(tensor_data));
  blueoil::Tensor tensor2({3, 2}, reinterpret_cast<float*>(tensor_data2));  // shape diff
  blueoil::Tensor tensor3({2, 3}, reinterpret_cast<float*>(tensor_data3));  // data diff

  float *arr = tensor0.dataAsArray({1, 0});
  if ((arr[0] != 7) || (arr[1] != 8) || (arr[2] != 9)) {
    std::cerr << "tensor0: at(1, 0) != {7, 8, 9}" << std::endl;
    tensor0.dump();
    return EXIT_FAILURE;
  }
  // equal
  if ((!tensor0.allequal(tensor1)) || (!tensor0.allclose(tensor1))) {
    std::cerr << "tensor_test: tensor0 != tensor1" << std::endl;
    tensor0.dump();
    tensor1.dump();
    return EXIT_FAILURE;
  }
  // shape different
  if (tensor1.allequal(tensor2) || tensor1.allclose(tensor2)) {
    std::cerr << "tensor_test: tensor1 == tensor2" << std::endl;
    tensor1.dump();
    tensor2.dump();
    return EXIT_FAILURE;
  }
  // data different
  if (tensor1.allequal(tensor3) || tensor1.allclose(tensor3)) {
    std::cerr << "tensor_test: tensor1 == tensor3" << std::endl;
    tensor1.dump();
    tensor3.dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


int main(void) {
  int status_code = test_tensor();
  std::exit(status_code);
}


/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

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

#include <string>
#include <iostream>
#include <vector>

#include "blueoil.hpp"

// TODO: delete this func. it is for debug.
blueoil::Tensor RandomImage(int height, int width, int channel) {
    blueoil::Tensor t({height, width, channel});
    std::vector<float>& data = t.data();
    unsigned seed = 0;

    for (size_t i = 0; i < data.size(); ++i) {
        const float f_rand = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX) * 255;
        data[i] = f_rand;
    }

    return t;
}



int main() {
  std::string meta_yaml = "meta.yaml";

  blueoil::Predictor predictor = blueoil::Predictor(meta_yaml);

  std::cout << "classes: " << std::endl;
  for (std::string j : predictor.classes) {
    std::cout << j << "\n";
  }

  std::cout << "task: " << predictor.task << std::endl;

  std::cout << "expected input shape: " << std::endl;
  for (int j : predictor.expected_input_shape) {
    std::cout << j << "\n";
  }

  blueoil::Tensor random_image = RandomImage(256, 256, 3);

  std::cout << "Run" << std::endl;
  blueoil::Tensor output =  predictor.Run(random_image);

  std::cout << "Results !" << std::endl;
  for (float j : output.data()) {
    std::cout << j << std::endl;
  }
}

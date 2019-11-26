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
#include <cstring>

#include "blueoil.hpp"


int main(int argc, char **argv) {
  std::string imagefile, meta_yaml;

  for (int i = 1; i < (argc-1); i++) {
    char *arg = argv[i];
    char *arg2 = argv[i+1];
    if ((arg[0] == '-') && (std::strlen(arg) == 2) && (arg2[0] != '-')) {
      switch (arg[1]) {
        case 'i':  // image file (ex. raw_image.npy)
          imagefile = std::string(arg2);
          break;
        case 'c':  // config file (meta.yaml)
          meta_yaml = std::string(arg2);
          break;
      }
    }
  }

  if (imagefile.empty() || meta_yaml.empty()) {
    std::cerr << "Usage: a.out -i <imagefile> -c <configfile>" << std::endl;
    std::cerr << "ex) a.out -i raw_image.npy -c meta.yaml" << std::endl;
    std::exit(1);
  }

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
  blueoil::Tensor image = blueoil::Tensor_loadImage(imagefile);

  std::cout << "Run" << std::endl;
  blueoil::Tensor output =  predictor.Run(image);

  std::cout << "Results !" << std::endl;
  output.dump();

  std::exit(0);
}

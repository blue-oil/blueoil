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
==============================================================================*/

#include <stdio.h>
#include <cstring>
#include <vector>

#include "global.h"
#include "dlk_test.h"
#include "network.h"
#include "time_measurement.h"
#include "npy.hpp"

template<typename T>
std::ostream& operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    if(!v.empty()) {
        std::copy(v.begin(), std::prev(v.end()), std::ostream_iterator<T>(os, "*"));
        os << v.back();
    }
    os << "]";

    return os;
}


int main(int argc, char *argv[])
{
  if(argc != 3)
  {
    std::cout << "Error: The number of arguments is invalid" << std::endl;
    std::cout << "Use: " << argv[0] << " <.npy debug input file> <.npy debug expected output file>" << std::endl;
    return 1;
  }

  std::vector<unsigned long> debug_input_shape;
  std::vector<unsigned long> debug_output_shape;

  std::vector<{{ graph_input.dtype.cpptype() }}> debug_input_data;
  std::vector<{{ graph_output.dtype.cpptype()}} > debug_output_data;

  try {
    npy::LoadArrayFromNumpy(argv[1], debug_input_shape, debug_input_data);
    npy::LoadArrayFromNumpy(argv[2], debug_output_shape, debug_output_data);
  }
  catch(std::exception &ex) {
    std::cout << "Unable to load the debug data: " << ex.what() << std::endl;
    return -1;
  }

  if({{ graph_input.view.size_in_words_as_cpp }} != debug_input_data.size()) {
    std::cout << "Error: debug input shape should be {{ graph_input.view.size_in_words_as_cpp }} but got " << debug_input_shape << std::endl;
    return -1;
  }

  if({{ graph_output.view.size_in_words_as_cpp }} != debug_output_data.size()) {
    std::cout << "Error: debug output shape should be {{ graph_output.view.size_in_words_as_cpp }} but got " << debug_output_shape << std::endl;
    return -1;
  }

  Measurement::Start("TotalInitTime");
  Network nn;
  bool initialized = nn.init();
  Measurement::Stop();

  if(!initialized) {
    std::cout << "Error: cannot initialize the network" << std::endl;
    return 1;
  }

  std::vector<{{ graph_output.dtype.cpptype() }}> output({{ graph_output.view.size_in_words_as_cpp }});

  Measurement::Start("TotalRunTime");
  nn.run(debug_input_data.data(), output.data());
  Measurement::Stop();

  bool test_result = dlk_test::compare(output.data(), "Default network test ", debug_output_data.data(), output.size());

  Measurement::Report();

  return test_result;
}

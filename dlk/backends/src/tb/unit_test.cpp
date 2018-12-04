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

#include "common/global.h"
#include "tb/test_a8w1_conv.h"
#include "tb/test_conv.h"

using std::cout;
using std::endl;

bool parse_input_type(int argc, char const *argv[], input_type &in_type)
{
  bool flag = true;

  if (argc != 2) {
    cout << "Error: 1st arg " << argv[0] << " is not supported..." << endl;
    cout << "Available input type: <sequential|random|all_1>" << endl;
    flag = false;
  }

  if (std::strcmp(argv[1], "sequential") == 0) {
    in_type = SEQUENTIAL;
  } else if (std::strcmp(argv[1], "random") == 0) {
    in_type = RANDOM;
  } else if (std::strcmp(argv[1], "all_1") == 0) {
    in_type = ALL_1;
  } else {
    cout << "Error: input type is not supported." << endl << "Available input type: <sequential|random|add_1>" << endl;
    flag = false;
  }

  return flag;
}

bool test_conv(input_type &in_type)
{
  srand((unsigned int)time(NULL));

  bool res = true;

  // test conv1x1
  Conv_params_t conv1x1_p = new_Conv_params(conv1x1_params) res &= test_conv<1, 1>(in_type, conv1x1_p);

  // test conv3x3
  Conv_params_t conv3x3_p = new_Conv_params(conv3x3_params) res &= test_conv<3, 3>(in_type, conv3x3_p);

  return res;
}

bool test_a8w1_conv(input_type &in_type)
{
  srand((unsigned int)time(NULL));

  bool res = test_a8w1_conv<3, 3>(in_type);
  return res;
}

int main(int argc, char const *argv[])
{
  input_type in_type;

  bool input_valid = parse_input_type(argc, argv, in_type);
  if (!input_valid) {
    return 1;
  }

  bool res_conv = true;
  res_conv &= test_conv(in_type);
  // res_conv &= test_a8w1_conv(in_type);

  return (res_conv) ? 0 : 1;
}

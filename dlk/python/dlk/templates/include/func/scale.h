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

#ifndef DLK_FUNC_SCALE_H_INCLUDED
#define DLK_FUNC_SCALE_H_INCLUDED

#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"

template <MemoryLayout layout>
void func_Scale(const TensorView<T_INT, layout>& input,
    const TensorView<T_FLOAT, MemoryLayout::Atom>& factor,
    const TensorView<T_FLOAT, layout>& output,
    unsigned input_bitwidth) {
  Measurement::Start("Scale");

  assert(input.get_shape() == output.get_shape());
  T_UINT elements = output.size();

  // e.g. 2 bits -> 4 - 1 -> 3
  T_FLOAT power_of_two_minus_one;

  if (input_bitwidth == 8) {
    power_of_two_minus_one = 255.0f;

    for (T_UINT i = 0; i < elements; i++) {
      output.data()[i] = (input.data()[i] / power_of_two_minus_one) * factor();
    }
  } else if (input_bitwidth == 2) {
    power_of_two_minus_one = 3.0f;
    T_FLOAT s = factor() / power_of_two_minus_one * 2.0f;

    for (T_UINT i = 0; i < elements; i++) {
      output.data()[i] = input.data()[i] * s;
    }
  } else {
    std::cout << "2 nor 8 Convolution: Not Implemented Yet!!\n" << std::endl;
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_SCALE_H_INCLUDED

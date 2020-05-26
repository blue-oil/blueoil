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

#ifndef QUANTIZER_H_INCLUDED
#define QUANTIZER_H_INCLUDED

#include <cmath>

#include "global.h"
#include "tensor_view.h"

template <MemoryLayout layout>
void func_QTZ_binary_mean_scaling(
    const TensorView<T_FLOAT, layout>& input,
    const TensorView<T_FLOAT, layout>& output) {
  T_FLOAT sum = 0.f;
  unsigned num_elems = input.size();

  for(unsigned i = 0; i < num_elems; i++)
  {
    sum += std::abs(input[i]);
  }

  T_FLOAT mean = sum / num_elems;
  T_FLOAT mean_minus = -1 * mean;

  for(unsigned i = 0; i < num_elems; i++)
  {
    output.data()[i] = (input.data()[i] >= 0) ? mean : mean_minus;
  }
}

void func_LinearMidTreadHalfQuantizer(
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_INT, MemoryLayout::Atom>& nbit,
    const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    BYTE *temporary_buf);

void func_LinearMidTreadHalfQuantizer(
  const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
  const TensorView<T_INT, MemoryLayout::Atom>& nbit,
  const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
  const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
  BYTE *temporary_buf);

#endif // QUANTIZER_H_INCLUDED

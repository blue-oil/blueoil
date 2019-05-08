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

#ifndef DLK_FUNC_EXTRACT_IMAGE_PATCHES
#define DLK_FUNC_EXTRACT_IMAGE_PATCHES

#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"
#include "pack_input_to_qwords.h"
#include <limits.h>

template<class T>
void func_ExtractImagePatches(
    const TensorView<T, MemoryLayout::NHWC>& input,
    const TensorView<T, MemoryLayout::NHWC>& output,
    T_UINT kernel_size, T_UINT stride)
{
  Measurement::Start("ExtractImagePatches");
  const auto in_shape = input.get_shape();
  const T_UINT input_width = in_shape[2];
  const T_UINT input_depth = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[1];
  const T_UINT out_width = out_shape[2];
  const T_UINT out_depth = out_shape[3];

  const int bits_per_input = 2;
  const int bits_per_word = sizeof(QUANTIZED_PACKED) * CHAR_BIT;
  int full_words_in_depth = input_depth / bits_per_word;
  int remainder_bits_in_depth = input_depth % bits_per_word;

  T_UINT packed_input_depth = full_words_in_depth * bits_per_input + (remainder_bits_in_depth ? bits_per_input : 0);
  T_UINT packed_output_size = packed_input_depth * kernel_size * kernel_size * out_width * out_height;

  auto output_buffer = std::make_unique<QUANTIZED_PACKED[]>(packed_output_size);

  auto* out = (input_depth < 32) ? output_buffer.get() : output.data();
  T_UINT output_index = 0;
  for(T_UINT wi = 0; wi < out_height; wi++)
    for(T_UINT wj = 0; wj < out_width; wj++)
    {
      for(T_UINT ki = 0; ki < kernel_size; ki++)
        for(T_UINT kj = 0; kj < kernel_size; kj++)
          for(T_UINT kz = 0; kz < packed_input_depth; kz++)
          {
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;

            out[output_index++] = input(0, row, col, kz);
          }
    }

  if(input_depth < 32) {
    T_UINT chunk_size = input_depth;
    T_UINT chunks_per_word = 32 / chunk_size;
    T_UINT chunk_mask = (1 << chunk_size) - 1;

    for(int offset = 0;  offset < bits_per_input; offset++) {
        T_UINT current_input_word = offset;
        T_UINT current_output_word = offset;

        for(int output_blocks = 0;  output_blocks < (packed_output_size / bits_per_input); output_blocks += chunks_per_word) {
          output.data()[current_output_word] = QUANTIZED_PACKED(0);
          for(int chunk = 0; chunk < chunks_per_word; chunk++) {
            output.data()[current_output_word] = QUANTIZED_PACKED(output.data()[current_output_word].Raw() | (output_buffer[current_input_word].Raw() & chunk_mask) << (chunk * chunk_size));
            current_input_word += bits_per_input;
          }
          current_output_word += bits_per_input;
        }
    }
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_EXTRACT_IMAGE_PATCHES

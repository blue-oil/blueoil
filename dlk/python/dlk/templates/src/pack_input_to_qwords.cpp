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
#include <limits.h>
#include "global.h"
#include "pack_input_to_qwords.h"
#include "pack2bits.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

int pack_input(QUANTIZED_NOT_PACKED input[], size_t input_height, size_t input_width, size_t input_depth,
  size_t bits_per_input, QUANTIZED_PACKED output[]) {

  Measurement::Start("pack_input");
  const int bits_per_word = sizeof(QUANTIZED_PACKED) * CHAR_BIT;
  int full_words_in_depth = input_depth / bits_per_word;
  int remainder_bits_in_depth = input_depth % bits_per_word;
  int input_index = 0;
  int current_word = 0;

  auto len = input_height * input_width * input_depth;
  if (input_depth % 32 == 0) {
      pack2bits(input, output, (int)len);
      Measurement::Stop();
      return 0;
  }

  for (int h = 0; h < input_height; ++h)
      for (int w = 0; w < input_width; ++w) {
          for (int d = 0; d < full_words_in_depth; ++d) {
              output[current_word] = QUANTIZED_PACKED(0);
              output[current_word + 1] = QUANTIZED_PACKED(0);
              for (int d = 0; d < bits_per_word; ++d) {
                for(int b = 0; b < bits_per_input; ++b)
                  output[current_word + b] = QUANTIZED_PACKED(output[current_word + b].Raw() | ((input[input_index] >> b) & 1) << d);
                input_index++;
              }
              current_word += bits_per_input;
          }

          if(!remainder_bits_in_depth)
              continue;

          output[current_word] = QUANTIZED_PACKED(0);
          output[current_word + 1] = QUANTIZED_PACKED(0);
          for (int d = 0; d < remainder_bits_in_depth; ++d) {
             for(int b = 0; b < bits_per_input; ++b)
               output[current_word + b] = QUANTIZED_PACKED(output[current_word + b].Raw() | ((input[input_index] >> b) & 1) << d);
             input_index++;
          }
          current_word += bits_per_input;
      }

  Measurement::Stop();

  return current_word;
}

void pack_input_to_qwords(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED output[],
  struct binary_convolution_parameters bcp)
{
  struct convolution_parameters& p = bcp.normal_conv_params;
  unsigned kernel_elems = p.kernel_height * p.kernel_width * p.kernel_depth;
  unsigned im2col_input_elems = p.output_height * p.output_width * kernel_elems;

  pack_input(input, bcp.normal_conv_params.input_height, bcp.normal_conv_params.input_width,
    bcp.normal_conv_params.kernel_depth, bcp.bin_input_bitwidth, output);
}

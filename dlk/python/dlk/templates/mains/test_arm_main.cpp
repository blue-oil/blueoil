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

#include <cstring>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <random>

#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "pack_input_to_qwords.h"
#include "time_measurement.h"

// old function just for testing the current implementation
void old_pack_input_to_qwords(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED output[],
  struct binary_convolution_parameters bcp)
{
  struct convolution_parameters p = bcp.normal_conv_params;

  unsigned kernel_elems = p.kernel_height * p.kernel_width * p.kernel_depth;
  unsigned im2col_input_elems = p.output_height * p.output_width * kernel_elems;
  const unsigned nbit_qinput_word = sizeof(QUANTIZED_PACKED) * 8;

  unsigned idx_out = 0;
  unsigned bit_count = 0;
  unsigned total_bit_count = 0;

  static QUANTIZED_PACKED qinput_words_buf[MAX_NBIT_QINPUT] = {}; // must be initialized

  for (unsigned idx_in = 0; idx_in < im2col_input_elems; idx_in++)
  {
    QUANTIZED_NOT_PACKED tmp_input = input[idx_in];

    for (unsigned i_bit = 0; i_bit < bcp.bin_input_bitwidth; i_bit++)
    {
      QUANTIZED_NOT_PACKED tmp_input_bit = tmp_input & 0x1;
      qinput_words_buf[i_bit] |= (tmp_input_bit << bit_count);
      tmp_input = (tmp_input >> 1);
    }

    ++bit_count;
    ++total_bit_count;

    if (bit_count == nbit_qinput_word || total_bit_count == kernel_elems)
    {
      for (unsigned i_bit = 0; i_bit < bcp.bin_input_bitwidth; i_bit++) {
        output[idx_out++] = qinput_words_buf[i_bit];
      }

      bit_count = 0;
      std::memset(qinput_words_buf, 0, MAX_NBIT_QINPUT * sizeof(QUANTIZED_PACKED));
    }
  }
}

void generate_random_input(QUANTIZED_NOT_PACKED array[], int size) {
  std::random_device rnd;
  int max = 1 << (sizeof(QUANTIZED_NOT_PACKED) * 8);

  for (int i = 0 ; i < size ; i++) {
    array[i] = (QUANTIZED_NOT_PACKED)(rnd() % max);
  }
}

bool random_test_and_compare(int num_input_elems, int num_output_elems, binary_convolution_parameters bcp) {
  // set up input and outputs
  QUANTIZED_NOT_PACKED input1[num_input_elems];
  QUANTIZED_NOT_PACKED input2[num_input_elems];
  generate_random_input(input1, sizeof input1);
  std::memcpy(input2, input1, sizeof input1);
  QUANTIZED_PACKED output1[num_output_elems];
  QUANTIZED_PACKED output2[num_output_elems];

  // pack
  pack_input_to_qwords(input1, output1, bcp);
  old_pack_input_to_qwords(input2, output2, bcp);

  // compare
  int result = std::memcmp(output1, output2, num_output_elems * sizeof(QUANTIZED_PACKED));
  return result == 0;
}

int main(int argc, char *argv[])
{
  // setup
  struct binary_convolution_parameters bcp;
  struct convolution_parameters *p = &(bcp.normal_conv_params);
  p->kernel_height = 3;
  p->kernel_width = 3;
  p->kernel_depth = 32;
  p->input_height = 40;
  p->input_width = 40;
  p->output_height = 40;
  p->output_width = 40;
  bcp.bin_input_bitwidth = 2;
  int num_input_elems = p->kernel_height * p->kernel_width * p->kernel_depth * p->output_height * p->output_width;
  int num_output_elems = (num_input_elems / 4) / (sizeof(QUANTIZED_PACKED) / sizeof(QUANTIZED_NOT_PACKED));

  // set up input and outputs
  QUANTIZED_NOT_PACKED input1[num_input_elems];
  QUANTIZED_NOT_PACKED input2[num_input_elems];
  generate_random_input(input1, sizeof input1);
  std::memcpy(input2, input1, sizeof input1 * sizeof(QUANTIZED_NOT_PACKED));
  
  // check if the two input arrays are the same
  if (std::memcmp(input1, input2, sizeof input1 * sizeof(QUANTIZED_NOT_PACKED)) == 0)
    std::cout << "input set up" << std::endl;
  else {
    std::cerr << "failed in copying inputs." << std::endl;
    exit(-1);
  }
    

  QUANTIZED_PACKED output1[num_output_elems];
  QUANTIZED_PACKED output2[num_output_elems];

  bool result = true;
  for (int i = 0 ; i < 100 ; i++) {
    result = random_test_and_compare(num_input_elems, num_output_elems, bcp);
    if (!result) {
      std::cerr << "Failed at " << i << "th iteration." << std::endl;
      break;
    }
    else if ((i + 1) % 10 == 0)
      std::cout << "Passed " << i + 1 << " tests." << std::endl;
  }

  if (result)
    std::cout << "Succeeded in testing pack_input_to_qwords()" << std::endl;

  return result ? 0 : -1;
}

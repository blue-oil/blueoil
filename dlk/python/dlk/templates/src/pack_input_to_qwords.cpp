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

#include "global.h"
#include "pack_input_to_qwords.h"
#include "pack2bits.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

void pack_input_to_qwords(QUANTIZED_NOT_PACKED input[],
                          QUANTIZED_PACKED output[],
                          unsigned int len,
                          unsigned int input_bitwidth)
{
  Measurement::Start("pack_input_to_qwords");

  #ifdef USE_NEON
    if (len % 16 == 0) {
      pack2bits(input, output, (int)len);
      Measurement::Stop();
      return;
    }
  #endif

  unsigned idx_in = 0;
  unsigned idx_out = 0;
  unsigned bit_count = 0;

  static QUANTIZED_PACKED::T qinput_words_buf[MAX_NBIT_QINPUT] = {}; // must be initialized

  for (idx_in = 0; idx_in + 3 < len; idx_in += 4) {
    QUANTIZED_PACKED::T t = ((QUANTIZED_PACKED::T*)(&input[idx_in]))[0];
    QUANTIZED_PACKED::T b0 = (t & 0x00000001)
      | ((t & 0x01000000) >> 21)
      | ((t & 0x00010000) >> 14)
      | ((t & 0x00000100) >> 7);
    QUANTIZED_PACKED::T b1 = ((t & 0x02000000) >> 22)
      | ((t & 0x00020000) >> 15)
      | ((t & 0x00000200) >> 8)
      | ((t & 0x00000002) >> 1);

    qinput_words_buf[0] |= (b0 << bit_count);
    qinput_words_buf[1] |= (b1 << bit_count);
    bit_count += 4;

    if (bit_count == QUANTIZED_PACKED::BitCount)
      {
        for (unsigned i_bit = 0; i_bit < input_bitwidth; i_bit++) {
          output[idx_out++] = QUANTIZED_PACKED(qinput_words_buf[i_bit]);
          qinput_words_buf[i_bit] = 0;
        }

        bit_count = 0;
      }
  }

  // Remainder
  {
    for (; idx_in < len; idx_in++) {
      QUANTIZED_NOT_PACKED tmp_input = input[idx_in];
      QUANTIZED_PACKED::T b0 = tmp_input & 0x1;
      QUANTIZED_PACKED::T b1 = (tmp_input & 0x2) >> 1;

      qinput_words_buf[0] |= (b0 << bit_count);
      qinput_words_buf[1] |= (b1 << bit_count);
      ++bit_count;
    }

    for (unsigned i_bit = 0; i_bit < input_bitwidth; i_bit++) {
      output[idx_out++] = QUANTIZED_PACKED(qinput_words_buf[i_bit]);
      // FIX ME (cannot delete below line now,
      // later change implementation to use two register because input_bitwidth is always 2.)
      qinput_words_buf[i_bit] = 0;
    }
  }

  Measurement::Stop();
}


void pack_input_to_qwords(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED output[],
  struct binary_convolution_parameters bcp)
{
  struct convolution_parameters& p = bcp.normal_conv_params;
  unsigned kernel_elems = p.kernel_height * p.kernel_width * p.kernel_depth;
  unsigned im2col_input_elems = p.output_height * p.output_width * kernel_elems;
  pack_input_to_qwords(input, output, im2col_input_elems, bcp.bin_input_bitwidth);
}

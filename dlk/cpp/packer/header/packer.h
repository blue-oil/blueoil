/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

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

#pragma once


#include <iostream>
#include <fstream>


class Packer
{
public:
  Packer() :
    bitwidth(1),
    kernel_size(0),
    nkernels(0),
    wordsize(32),
    extra_bits(0),
    words_per_patch(0),
    mux_mode(1),
    extra_bit_value(true)
  {}

  void set_bitwidth(uint32_t bitwidth)
  {
    this->bitwidth = bitwidth;
  }

  void set_wordsize(uint32_t wordsize)
  {
    this->wordsize = wordsize;
  }

  void set_multiplexing_mode(uint32_t mode)
  {
    this->mux_mode = mode;
  }

  void set_extra_bit_value(bool value)
  {
    this->extra_bit_value = value;
  }

  uint32_t get_wordsize()
  {
    return this->wordsize;
  }

  uint32_t get_output_size(uint32_t size, uint32_t number_of_kernels)
  {
    uint32_t bitwidth_per_weight = 0;
    uint32_t nword_multiplier = 0;

    switch(mux_mode)
    {
      case 1: // Mux sequential: one word contains many channels
      {
        bitwidth_per_weight = bitwidth;
        nword_multiplier = 1;
        break;
      }

      case 2: // Mux sequential interleaving words: alternating channels on every word
      case 3: // Mux sequential: whole first channel followed by the whole second channel and so on
      {
        bitwidth_per_weight = 1;
        nword_multiplier = bitwidth;
        break;
      }

      default:
      {
        return 0;
      }
    }

    nkernels = number_of_kernels;
    kernel_size = size / number_of_kernels;

    uint32_t kernel_bits = kernel_size * bitwidth_per_weight;
    uint32_t extra = kernel_bits % wordsize;

    extra_bits = (extra == 0 ? 0 : wordsize - extra);

    uint32_t bits_per_patch = kernel_bits + extra_bits;
    words_per_patch = (bits_per_patch / wordsize) + (bits_per_patch % wordsize > 0 ? 1 : 0);

    uint32_t nwords = words_per_patch * nword_multiplier * number_of_kernels;

    return nwords;
  }

  template<typename OTYPE>
  void run(float *input, uint32_t size, OTYPE *output)
  {
    uint32_t bit_index = 0;
    uint32_t weight_index = 0;
    uint32_t channel_group_offset = 0;
    uint32_t group_size = (mux_mode == 3 ? words_per_patch : 1);

    for(uint32_t k = 0; k < nkernels; k++)
    {
      for(uint32_t w = 0; w < kernel_size; w++)
      {
        uint32_t e = uint32_t(input[weight_index++]);

        for(uint32_t b = 0; b < bitwidth; b++)
        {
          uint32_t word_index = channel_group_offset + (mux_mode == 2 || mux_mode == 3 ? b * group_size : 0);

          if((e & 0x1) == 1)
            output[word_index] |=  (OTYPE(1) << (bit_index % wordsize));
          else
            output[word_index] &= ~(OTYPE(1) << (bit_index % wordsize));

          e >>= 1;
          bit_index += (mux_mode == 2 || mux_mode == 3 ? 0 : 1);
          channel_group_offset += (mux_mode == 1 && bit_index % wordsize == 0 ? 1 : 0);
        }

        bit_index += (mux_mode == 2 || mux_mode == 3 ? 1 : 0);
        channel_group_offset += (mux_mode == 2 && bit_index % wordsize == 0 ? bitwidth : 0);
        channel_group_offset += (mux_mode == 3 && bit_index % wordsize == 0 ? 1 : 0);
      }

      uint32_t bitwidth_extra = (mux_mode == 1 ? 1 : bitwidth);

      for(uint32_t e = 0; e < extra_bits; e++)
      {
        for(uint32_t b = 0; b < bitwidth_extra; b++)
        {
          uint32_t word_index = channel_group_offset + (mux_mode == 2 || mux_mode == 3 ? b * group_size: 0);

          if(extra_bit_value)
            output[word_index] |=  (OTYPE(1) << (bit_index % wordsize));
          else
            output[word_index] &= ~(OTYPE(1) << (bit_index % wordsize));

          bit_index += (mux_mode == 2 || mux_mode == 3 ? 0 : 1);
          channel_group_offset += (mux_mode == 1 && bit_index % wordsize == 0 ? 1 : 0);
        }

        bit_index += (mux_mode == 2 || mux_mode == 3 ? 1 : 0);
        channel_group_offset += (mux_mode == 2 && bit_index % wordsize == 0 ? bitwidth : 0);
        channel_group_offset += (mux_mode == 3 && bit_index % wordsize == 0 ? 1 : 0);
      }

      channel_group_offset += (mux_mode == 3 ? (bitwidth - 1) * group_size : 0);
    }
  }

private:
  uint32_t bitwidth;
  uint32_t kernel_size;
  uint32_t nkernels;
  uint32_t wordsize;
  uint32_t extra_bits;
  uint32_t words_per_patch;
  uint32_t mux_mode;

  bool extra_bit_value;
};

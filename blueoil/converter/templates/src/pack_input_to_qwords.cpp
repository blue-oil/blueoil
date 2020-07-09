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
#include <limits.h>
#include "global.h"
#include "pack_input_to_qwords.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"
#ifdef USE_NEON
#include <arm_neon.h>
#endif
#ifdef USE_AVX
#include <x86intrin.h>
#endif

void pack_input(QUANTIZED_NOT_PACKED input[], size_t input_height, size_t input_width, size_t input_channels,
  size_t bits_per_input, QUANTIZED_PACKED output[]) {

  Measurement::Start("pack_input");
  constexpr size_t bits_per_word = sizeof(QUANTIZED_PACKED) * CHAR_BIT;
  const size_t full_words_in_channels = input_channels / bits_per_word;
  const size_t blocks_in_channels = (input_channels + bits_per_word - 1) / bits_per_word;
  const size_t remainder_bits_in_channels = input_channels % bits_per_word;

  const auto area = input_height * input_width;
#ifdef USE_NEON
  if (bits_per_input == 2 && input_channels % bits_per_word == 0) {
    const uint8_t coeff_ary[16] = {
      1, 2, 4, 8, 16, 32, 64, 128,
      1, 2, 4, 8, 16, 32, 64, 128,
    };
    const auto coeff = vld1q_u8(coeff_ary);
    const auto vone = vdupq_n_u8(1);
#pragma omp parallel for
    for (size_t i = 0; i < area; ++i) {
      for (size_t j = 0; j < full_words_in_channels; ++j) {
        const auto v0 = vld1q_u8(input + i*blocks_in_channels*bits_per_word + j*bits_per_word +  0);
        const auto v1 = vld1q_u8(input + i*blocks_in_channels*bits_per_word + j*bits_per_word + 16);
        const auto l0 = vandq_u8(v0, vone);
        const auto l1 = vandq_u8(v1, vone);
        const auto m0 = vshrq_n_u8(v0, 1);
        const auto m1 = vshrq_n_u8(v1, 1);
        const auto ml0 = vmulq_u8(l0, coeff);
        const auto ml1 = vmulq_u8(l1, coeff);
        const auto mm0 = vmulq_u8(m0, coeff);
        const auto mm1 = vmulq_u8(m1, coeff);
        const auto al0 = vpadd_u8(vget_low_u8(ml0), vget_high_u8(ml0));
        const auto al1 = vpadd_u8(vget_low_u8(ml1), vget_high_u8(ml1));
        const auto am0 = vpadd_u8(vget_low_u8(mm0), vget_high_u8(mm0));
        const auto am1 = vpadd_u8(vget_low_u8(mm1), vget_high_u8(mm1));
        const auto bl = vpadd_u8(al0, al1);
        const auto bm = vpadd_u8(am0, am1);
        const auto c = vpadd_u8(bl, bm);
        vst1_u8(reinterpret_cast<uint8_t*>(output + i*bits_per_input + j*area*bits_per_input), c);
      }
    }
    Measurement::Stop();
    return;
  }
#endif

#ifdef USE_AVX
  if (bits_per_input == 2 && input_channels % bits_per_word == 0) {
#pragma omp parallel for
    for (size_t i = 0; i < area; ++i) {
      for (size_t j = 0; j < full_words_in_channels; ++j) {
        const auto a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(input + i*blocks_in_channels*bits_per_word + j*bits_per_word));
        const auto l = _mm256_movemask_epi8(_mm256_slli_epi16(a, 7));
        const auto m = _mm256_movemask_epi8(_mm256_slli_epi16(a, 6));
        output[i*bits_per_input + j*area*bits_per_input + 0] = QUANTIZED_PACKED(l);
        output[i*bits_per_input + j*area*bits_per_input + 1] = QUANTIZED_PACKED(m);
      }
    }
    Measurement::Stop();
    return;
  }
#endif

  for (size_t i = 0; i < area; ++i) {
    for (size_t j = 0; j < full_words_in_channels; ++j) {
      for (size_t b = 0; b < bits_per_input; ++b) {
        QUANTIZED_PACKED tmp(0);
        for (size_t d = 0; d < bits_per_word; ++d) {
          QUANTIZED_PACKED::base_t in = input[i*input_channels + j*bits_per_word + d];
          tmp |= QUANTIZED_PACKED(((in >> b) & 1) << d);
        }
        output[i*bits_per_input + j*area*bits_per_input + b] = tmp;
      }
    }

    if (!remainder_bits_in_channels)
      continue;

    for (size_t b = 0; b < bits_per_input; ++b) {
      QUANTIZED_PACKED tmp(0);
      for (size_t d = 0; d < remainder_bits_in_channels; ++d) {
        QUANTIZED_PACKED::base_t in = input[i*input_channels + full_words_in_channels*bits_per_word + d];
        tmp |= QUANTIZED_PACKED(((in >> b) & 1) << d);
      }
      output[i*bits_per_input + full_words_in_channels*area*bits_per_input + b] = tmp;
    }
  }

  Measurement::Stop();

  return;
}

void pack_input_to_qwords(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED output[],
  struct binary_convolution_parameters bcp)
{
  struct convolution_parameters& p = bcp.normal_conv_params;
  unsigned kernel_elems = p.kernel_height * p.kernel_width * p.input_channels;
  unsigned im2col_input_elems = p.output_height * p.output_width * kernel_elems;

  pack_input(input, bcp.normal_conv_params.input_height, bcp.normal_conv_params.input_width,
    bcp.normal_conv_params.input_channels, bcp.bin_input_bitwidth, output);
}

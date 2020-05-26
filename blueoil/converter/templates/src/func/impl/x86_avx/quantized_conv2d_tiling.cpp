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

#include <cassert>
#include <climits>
#include <limits>

#include "global.h"
#include "func/impl/quantized_conv2d_tiling.h"
#include "time_measurement.h"

#include <x86intrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dlk {

namespace impl {

void pack_input_for_tiling(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& input,
    const tiling_input_t& output) {
  Measurement::Start("Pack_input_for_tiling");
  const std::size_t in_channels = input.get_shape()[3];
  const std::size_t in_height = input.get_shape()[1];
  const std::size_t in_width = input.get_shape()[2];
  const std::size_t in_bitwidth = output.get_shape()[3];
  
  constexpr std::size_t InTypeBitWidth = CHAR_BIT * sizeof(uint32_t);
  const std::size_t in_stride = (in_channels + InTypeBitWidth - 1) / InTypeBitWidth;
#pragma omp parallel for schedule(dynamic)
  for (std::size_t in_ch_high = 0; in_ch_high < in_stride; ++in_ch_high) {
    for (std::size_t row = 0; row < in_height; ++row) {
      for (std::size_t col = 0; col < in_width; ++col) {
        for (std::size_t in_bit_ch = 0; in_bit_ch < in_bitwidth; ++in_bit_ch) {
          output(in_ch_high, row, col, in_bit_ch, 0) = tiling_input_elem_t(0);
        }
      }
    }
  }
#pragma omp parallel for schedule(dynamic)
  for (std::size_t row = 0; row < in_height; ++row) {
    for (std::size_t col = 0; col < in_width; ++col) {
      for (std::size_t in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
        for (std::size_t in_ch_low = 0; in_ch_low < InTypeBitWidth; ++in_ch_low) {
          std::size_t in_ch = in_ch_high + in_ch_low;
          if (in_ch >= in_channels) break;
          QUANTIZED_NOT_PACKED val = input(0, row, col, in_ch);
          for (std::size_t in_bit_ch = 0; in_bit_ch < in_bitwidth; ++in_bit_ch) {
            tiling_input_elem_base_t bit = (val >> in_bit_ch) & 1;
            output(in_ch_high / InTypeBitWidth, row, col, in_bit_ch, 0) |= tiling_input_elem_t(bit << in_ch_low);
          }
        }
      }
    }
  }

  Measurement::Stop();
}

void convert_thresholds(BIN_CONV_OUTPUT *input, BIN_CONV_OUTPUT *output, std::size_t channels) {
  constexpr std::size_t b = tiling_input_elem_t::BitCount;
  const auto channels_padded = channels + (b - channels % b) % b;
  const auto table = _mm256_setr_epi8(
      0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
      0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
  );
  const auto table2 = _mm256_setr_epi32(
      0, 4, 2, 6, 1, 5, 3, 7
  );
  std::size_t i = 0;
  for (; i + 16 <= channels; i += 16) {
    const auto v0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(input + NUM_OF_A2W1_THRESHOLD * i +  0));
    const auto v1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(input + NUM_OF_A2W1_THRESHOLD * i + 16));
    const auto v2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(input + NUM_OF_A2W1_THRESHOLD * i + 32));
    const auto v3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(input + NUM_OF_A2W1_THRESHOLD * i + 48));
    const auto tmp00 = _mm256_shuffle_epi8(v0, table);
    const auto tmp01 = _mm256_shuffle_epi8(v1, table);
    const auto tmp02 = _mm256_shuffle_epi8(v2, table);
    const auto tmp03 = _mm256_shuffle_epi8(v3, table);
    const auto tmp10 = _mm256_unpacklo_epi32(tmp00, tmp01);
    const auto tmp11 = _mm256_unpacklo_epi32(tmp02, tmp03);
    const auto tmp12 = _mm256_unpackhi_epi32(tmp00, tmp01);
    const auto tmp13 = _mm256_unpackhi_epi32(tmp02, tmp03);
    const auto tmp20 = _mm256_unpacklo_epi32(tmp10, tmp11);
    const auto tmp21 = _mm256_unpackhi_epi32(tmp10, tmp11);
    const auto tmp22 = _mm256_unpacklo_epi32(tmp12, tmp13);
    const auto tmp23 = _mm256_unpackhi_epi32(tmp12, tmp13);
    const auto th0 = _mm256_permutevar8x32_epi32(tmp20, table2);
    const auto th1 = _mm256_permutevar8x32_epi32(tmp21, table2);
    const auto th2 = _mm256_permutevar8x32_epi32(tmp22, table2);
    const auto flg = _mm256_permutevar8x32_epi32(tmp23, table2);
    const auto is_neg = _mm256_cmpgt_epi16(_mm256_setzero_si256(), flg);
    const auto res0 = _mm256_sub_epi16(th0, is_neg);
    const auto res1 = _mm256_sub_epi16(th1, is_neg);
    const auto res2 = _mm256_sub_epi16(th2, is_neg);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + 0 * channels_padded + i), res0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + 1 * channels_padded + i), res1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + 2 * channels_padded + i), res2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + 3 * channels_padded + i), flg);
  }
  for (; i < channels; ++i) {
    BIN_CONV_OUTPUT v0 = input[NUM_OF_A2W1_THRESHOLD * i + 0];
    BIN_CONV_OUTPUT v1 = input[NUM_OF_A2W1_THRESHOLD * i + 1];
    BIN_CONV_OUTPUT v2 = input[NUM_OF_A2W1_THRESHOLD * i + 2];
    const BIN_CONV_OUTPUT flg = input[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flg < 0) {
      --v0; --v1; --v2;
    }
    output[channels_padded * 0 + i] = v0;
    output[channels_padded * 1 + i] = v1;
    output[channels_padded * 2 + i] = v2;
    output[channels_padded * 3 + i] = flg;
  }
}

void QuantizedConv2DTiling(const tiling_input_t& input,
    const tiling_kernel_t& kernel,
    const binary_convolution_parameters &p) {
  constexpr std::size_t InTypeBitWidth = tiling_input_elem_t::BitCount;
  convolution_parameters cp = p.normal_conv_params;
  const std::size_t out_channels = cp.output_channels;
  const std::size_t kh = cp.kernel_height;
  const std::size_t kw = cp.kernel_width;
  const std::size_t in_bitwidth = 2;
  const std::size_t in_channels = cp.kernel_depth;
  const std::size_t in_height = cp.input_height;
  const std::size_t in_width = cp.input_width;
  const std::size_t in_stride = (in_channels + InTypeBitWidth - 1) / InTypeBitWidth;
  const std::size_t padding = cp.padding;
  const std::size_t out_height = cp.output_height;
  const std::size_t out_width = cp.output_width;
  const std::size_t out_size = out_height * out_width * out_channels;
  const std::size_t maxa = (1 << in_bitwidth) - 1;

  assert(kh * kw < 32);
  assert(in_channels * kh * kw * maxa <= std::numeric_limits<BIN_CONV_OUTPUT>::max());
  assert(in_height * in_width == out_height * out_width);
  assert((in_channels % InTypeBitWidth) == 0);

  Measurement::Start("Quantized Conv2D Tiling");
  if (p.thresholds != nullptr) {
  }

  if (kh == 1 && kw == 1) {
    constexpr std::size_t InChUnroll = InTypeBitWidth; // hardcoded, not configurable
    constexpr std::size_t OutChUnroll = 16; // hardcoded, not configurable
    constexpr std::size_t OutChUnroll2 = 32; // hardcoded, not configurable
    constexpr std::size_t OutChBlocks = OutChUnroll2 / OutChUnroll;
    constexpr std::size_t InBitChUnroll = 2; // hardcoded, not configurable
    constexpr std::size_t ColUnroll = 4; // hardcoded, not configurable
    const auto row_tile_count = in_height;
    const auto col_tile_count = (in_width + ColUnroll - 1) / ColUnroll;
    const auto total_tile_count = row_tile_count * col_tile_count;
    alignas(32) int16_t nksum_ary[MAX_IN_C];
    alignas(32) uint32_t nk[MAX_SIZE_QKERNELS_PER_LAYER];
    for (std::size_t i = 0; i < out_channels; ++i) {
      nksum_ary[i] = 0;
    }
    for (std::size_t i = 0; i < out_channels; i += OutChUnroll) {
      for (std::size_t j = 0; j < in_channels / InTypeBitWidth; ++j) {
        for (std::size_t k = 0; k < OutChUnroll; ++k) {
          const auto nk_tmp
            = kernel.data()[(i+k) * (in_channels / InTypeBitWidth) + j].Raw();
          const auto nk_index = i * (in_channels / InTypeBitWidth) * 2
              + j * OutChUnroll * 2
              + k * 2;
          nk[nk_index + 0] = nk_tmp;
          nk[nk_index + 1] = nk_tmp;
          nksum_ary[i + k]
            += __builtin_popcount(nk_tmp) * 3;
        }
      }
    }
    const auto mask4 = _mm256_set1_epi8(0x0F);
    const auto vone = _mm256_set1_epi8(1);
    const auto popc_table = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
#pragma omp parallel for schedule(guided)
    for (std::size_t tile_index = 0; tile_index < total_tile_count; ++tile_index) {
      const auto col = tile_index % col_tile_count * ColUnroll;
      const auto row = tile_index / col_tile_count;
      alignas(32) uint32_t in_buf[MAX_IN_C/InTypeBitWidth][ColUnroll][2];
      for (std::size_t in_ch_high = 0; in_ch_high < in_channels/InTypeBitWidth; ++in_ch_high) {
        const auto in_index = in_ch_high * in_height * in_width * in_bitwidth
          + row * in_width * in_bitwidth
          + col * in_bitwidth;
        for (std::size_t j = 0; j < ColUnroll; ++j) {
          if (col + j >= in_width) {
            in_buf[in_ch_high][j][0] = 0;
            in_buf[in_ch_high][j][1] = 0;
          } else {
            in_buf[in_ch_high][j][0] = input.data()[in_index + j * 2 + 0].Raw();
            in_buf[in_ch_high][j][1] = input.data()[in_index + j * 2 + 1].Raw();
          }
        }
      }
      for (std::size_t Oh = 0; Oh < out_channels; Oh += OutChUnroll) {
        auto xnorsum0 = _mm256_setzero_si256();
        auto xnorsum1 = _mm256_setzero_si256();
        auto xnorsum2 = _mm256_setzero_si256();
        auto xnorsum3 = _mm256_setzero_si256();
        for (std::size_t in_ch_high = 0; in_ch_high < in_channels/InTypeBitWidth; ++in_ch_high) {
          const auto nk_index = Oh * (in_channels / InTypeBitWidth) * 2
            + in_ch_high * OutChUnroll * 2;
          const auto nk0 = _mm256_load_si256(reinterpret_cast<__m256i*>(&nk[nk_index +  0 * 2]));
          const auto nk1 = _mm256_load_si256(reinterpret_cast<__m256i*>(&nk[nk_index +  4 * 2]));
          const auto nk2 = _mm256_load_si256(reinterpret_cast<__m256i*>(&nk[nk_index +  8 * 2]));
          const auto nk3 = _mm256_load_si256(reinterpret_cast<__m256i*>(&nk[nk_index + 12 * 2]));
#define BINDP(i, j) \
  const auto xnor##j = in ^ nk##j; \
  const auto l4##j = mask4 & xnor##j; \
  const auto popc_l4##j = _mm256_shuffle_epi8(popc_table, l4##j); \
  const auto h4##j = mask4 & _mm256_srli_epi32(xnor##j, 4); \
  const auto popc_h4##j = _mm256_shuffle_epi8(popc_table, h4##j); \
  const auto cnt##j = _mm256_add_epi8(popc_l4##j, popc_h4##j); \
  const auto cnt16_##j = _mm256_maddubs_epi16(cnt##j, vone);

#define BINCONV(i) \
  do { \
    const auto in = _mm256_set1_epi64x(*reinterpret_cast<uint64_t*>(&in_buf[in_ch_high][i][0])); \
    BINDP(i, 0); \
    BINDP(i, 1); \
    const auto pack01 = _mm256_packs_epi16(cnt16_0, cnt16_1); \
    const auto cnt32_01 = _mm256_maddubs_epi16(pack01, vone); \
    BINDP(i, 2); \
    BINDP(i, 3); \
    const auto pack23 = _mm256_packs_epi16(cnt16_2, cnt16_3); \
    const auto cnt32_23 = _mm256_maddubs_epi16(pack23, vone); \
    const auto pack03 = _mm256_packs_epi16(cnt32_01, cnt32_23); \
    const auto cnt64 = _mm256_maddubs_epi16(pack03, _mm256_set1_epi16(0x0201)); \
    xnorsum##i = _mm256_add_epi16(xnorsum##i, cnt64); \
  } while(0)
          BINCONV(0);
          BINCONV(1);
          BINCONV(2);
          BINCONV(3);
#undef BINDP
#undef BINCONV
        }
        const auto nksum = _mm256_load_si256(reinterpret_cast<__m256i*>(nksum_ary + Oh));
        const auto table = _mm256_setr_epi32(
            0, 4, 1, 5, 2, 6, 3, 7
        );
        const auto permed0 = _mm256_permutevar8x32_epi32(xnorsum0, table);
        const auto permed1 = _mm256_permutevar8x32_epi32(xnorsum1, table);
        const auto permed2 = _mm256_permutevar8x32_epi32(xnorsum2, table);
        const auto permed3 = _mm256_permutevar8x32_epi32(xnorsum3, table);
        const auto ans0 = _mm256_sub_epi16(permed0, nksum);
        const auto ans1 = _mm256_sub_epi16(permed1, nksum);
        const auto ans2 = _mm256_sub_epi16(permed2, nksum);
        const auto ans3 = _mm256_sub_epi16(permed3, nksum);
        const auto Ohh = Oh / OutChUnroll2;
        const auto Om = Oh / OutChUnroll % OutChBlocks;
        if (p.thresholds != nullptr) {
          const auto th0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(p.thresholds + 0 * out_channels + Oh));
          const auto th1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(p.thresholds + 1 * out_channels + Oh));
          const auto th2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(p.thresholds + 2 * out_channels + Oh));
          const auto flg = _mm256_loadu_si256(reinterpret_cast<__m256i*>(p.thresholds + 3 * out_channels + Oh));
          const auto is_neg = _mm256_cmpgt_epi16(_mm256_setzero_si256(), flg);
          const auto m2 = _mm256_sub_epi16(flg, _mm256_set1_epi16(2));
          const auto is_not_const = _mm256_cmpgt_epi16(_mm256_setzero_si256(), m2);
#define APPLY_PACK(i) \
  if (col + i >= out_width) continue; \
  do { \
    const auto f0 = _mm256_andnot_si256(_mm256_cmpgt_epi16(th0, ans##i), flg); \
    const auto f1 = _mm256_andnot_si256(_mm256_cmpgt_epi16(th1, ans##i), flg); \
    const auto f2 = _mm256_andnot_si256(_mm256_cmpgt_epi16(th2, ans##i), flg); \
    const auto tmp = _mm256_add_epi16(_mm256_add_epi16(f0, f1), _mm256_add_epi16(f2, is_neg)); \
    const auto res = _mm256_blendv_epi8(m2, tmp, is_not_const); \
    const auto packed = _mm256_packs_epi16(res, _mm256_setzero_si256()); \
    const auto permed = _mm256_permute4x64_epi64(packed, 0xD8); \
    const auto shorted = _mm256_castsi256_si128(permed); \
    const auto vlsb = _mm_slli_epi16(shorted, 7); \
    const auto vmsb = _mm_slli_epi16(shorted, 6); \
    const auto lsb = _mm_movemask_epi8(vlsb); \
    const auto msb = _mm_movemask_epi8(vmsb); \
    reinterpret_cast<uint16_t*>(p.device_output_buf)[out_index + i * 2 * OutChBlocks + 0 * OutChBlocks] = lsb; \
    reinterpret_cast<uint16_t*>(p.device_output_buf)[out_index + i * 2 * OutChBlocks + 1 * OutChBlocks] = msb; \
  } while(0)
          const auto out_index = Ohh * out_height * out_width * 2 * OutChBlocks
              + row * out_width * 2 * OutChBlocks
              + col * 2 * OutChBlocks
              + Om;
          APPLY_PACK(0);
          APPLY_PACK(1);
          APPLY_PACK(2);
          APPLY_PACK(3);
        } else {
          auto out_buf = reinterpret_cast<BIN_CONV_OUTPUT*>(p.device_output_buf);
#define OUT(i) \
  if (col + i >= out_width) continue; \
  do { \
    const auto out_index = Ohh * out_height * out_width * OutChUnroll2 \
        + row * out_width * OutChUnroll2 \
        + (col + i) * OutChUnroll2 \
        + Om * OutChUnroll; \
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_buf + out_index), ans##i); \
  } while(0)
          OUT(0);
          OUT(1);
          OUT(2);
          OUT(3);
#undef OUT
        }
      }
    }
  } else {
    constexpr std::size_t InChUnroll = InTypeBitWidth; // hardcoded, not configurable
    constexpr std::size_t OutChUnroll = 8; // hardcoded, not configurable
    constexpr std::size_t OutChUnroll2 = 32; // hardcoded, not configurable
    constexpr std::size_t OutChBlocks = OutChUnroll2 / OutChUnroll;
    constexpr std::size_t InBitChUnroll = 2; // hardcoded, not configurable
    constexpr std::size_t ColUnroll = 3; // hardcoded, not configurable
    const std::size_t TileHeightMax = 20; // configurable
    const std::size_t TileWidthMax = 21; // configurable
    const std::size_t TileHeight = std::min(in_height, TileHeightMax);
    const std::size_t TileWidth = std::min(in_width + (ColUnroll - in_width % ColUnroll) % ColUnroll, TileWidthMax);
    const std::size_t khMax = 5;
    const std::size_t kwMax = 5;
    
    const std::size_t row_tile_count = (in_height + TileHeight - 1) / TileHeight;
    const std::size_t col_tile_count = (in_width + TileWidth - 1) / TileWidth;
    const std::size_t out_tile_count = (out_channels + OutChUnroll - 1) / OutChUnroll;
    const std::size_t total_tile_count = row_tile_count * col_tile_count * out_tile_count;
    const auto vone = _mm256_set1_epi8(0x01);
#pragma omp parallel for schedule(guided)
    for (std::size_t tile_index = 0; tile_index < total_tile_count; ++tile_index) {
      const auto out_ch_high = tile_index % out_tile_count;
      const auto col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
      const auto row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
      alignas(32) BIN_CONV_OUTPUT out_tile[TileHeightMax][TileWidthMax][OutChUnroll];
      for (std::size_t row = 0; row < TileHeight; ++row) {
        for (std::size_t col = 0; col < TileWidth; ++col) {
          for (std::size_t out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
            out_tile[row][col][out_ch] = 0;
          }
        }
      }
      const auto mask4 = _mm256_set1_epi8(0x0F);
      const auto vone = _mm256_set1_epi8(1);
      const auto popc_table = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
      );
      for (std::size_t in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
        alignas(32) QUANTIZED_PACKED_KERNEL notk[khMax][kwMax][OutChUnroll][2];
        alignas(32) BIN_CONV_OUTPUT notsum[OutChUnroll] = {};
        for (std::size_t out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          notsum[out_ch] = 0;
          for (std::size_t kr = 0; kr < kh; ++kr) {
            for (std::size_t kc = 0; kc < kw; ++kc) {
              const auto index = (out_ch_high * OutChUnroll + out_ch) * kh * kw * (in_channels / InTypeBitWidth)
                + kr * kw * (in_channels / InTypeBitWidth)
                + kc * (in_channels / InTypeBitWidth)
                + (in_ch_high / InTypeBitWidth);
              notk[kr][kc][out_ch][0] = kernel.data()[index];
              notk[kr][kc][out_ch][1] = kernel.data()[index];
              notsum[out_ch] += pop_count(notk[kr][kc][out_ch][0]) * 3;
            }
          }
        }
        for (std::size_t in_bit_ch_high = 0; in_bit_ch_high < in_bitwidth; in_bit_ch_high += InBitChUnroll) {
          alignas(32) tiling_input_elem_t in_tile[TileHeightMax + khMax - 1][TileWidthMax + kwMax - 1][InBitChUnroll];
          for (std::size_t row = 0; row < TileHeight + kh - 1; ++row) {
            if (row_high + row >= in_height + 2*padding) break;
            for (std::size_t col = 0; col < TileWidth + kw - 1; ++col) {
              if (col_high + col >= in_width + 2*padding) break;
              for (std::size_t in_bit_ch = 0; in_bit_ch < InBitChUnroll; ++in_bit_ch) {
                if (row_high + row < padding || row_high + row >= in_height + padding
                    || col_high + col < padding || col_high + col >= in_width + padding) {
                  in_tile[row][col][in_bit_ch] = tiling_input_elem_t(0);
                } else {
                  const auto index = (in_ch_high / InTypeBitWidth) * in_height * in_width * in_bitwidth
                    + (row_high + row - padding) * in_width * in_bitwidth
                    + (col_high + col - padding) * in_bitwidth
                    + (in_bit_ch_high + in_bit_ch);
                  in_tile[row][col][in_bit_ch] = input.data()[index];
                }
              }
            }
          }
          for (std::size_t row = 0; row < TileHeight; ++row) {
            for (std::size_t col = 0; col < TileWidth; col += ColUnroll) {
              auto xnorsum00 = _mm256_setzero_si256();
              auto xnorsum01 = _mm256_setzero_si256();
              auto xnorsum10 = _mm256_setzero_si256();
              auto xnorsum11 = _mm256_setzero_si256();
              auto xnorsum20 = _mm256_setzero_si256();
              auto xnorsum21 = _mm256_setzero_si256();
              for (std::size_t kr = 0; kr < kh; ++kr) {
                auto in0 = _mm256_set1_epi64x(*reinterpret_cast<uint64_t*>(&in_tile[row + kr][col + 0][0]));
                auto in1 = _mm256_set1_epi64x(*reinterpret_cast<uint64_t*>(&in_tile[row + kr][col + 1][0]));
                for (std::size_t kc = 0; kc < kw; ++kc) {
                  const auto nk0 = _mm256_load_si256(reinterpret_cast<__m256i*>(&notk[kr][kc][ 0][0]));
                  const auto nk1 = _mm256_load_si256(reinterpret_cast<__m256i*>(&notk[kr][kc][ 4][0]));
#define BINDP(i, j) \
  do { \
    const auto xnor = in##i ^ nk##j; \
    const auto l4 = mask4 & xnor; \
    const auto popc_l4 = _mm256_shuffle_epi8(popc_table, l4); \
    const auto h4 = mask4 & _mm256_srli_epi32(xnor, 4); \
    const auto popc_h4 = _mm256_shuffle_epi8(popc_table, h4); \
    const auto cnt = _mm256_add_epi8(popc_l4, popc_h4); \
    xnorsum##i##j = _mm256_add_epi8(xnorsum##i##j, cnt); \
  } while(0)

#define BINCONV(i) \
  do { \
    BINDP(i, 0); \
    BINDP(i, 1); \
  } while(0)
                  BINCONV(0);
                  BINCONV(1);
                  const auto in2 = _mm256_set1_epi64x(*reinterpret_cast<uint64_t*>(&in_tile[row + kr][col + kc + 2][0]));
                  BINCONV(2);
                  in0 = in1;
                  in1 = in2;
                }
              }
              const auto cnt16_00 = _mm256_maddubs_epi16(xnorsum00, vone);
              const auto cnt16_01 = _mm256_maddubs_epi16(xnorsum01, vone);
              const auto cnt16_10 = _mm256_maddubs_epi16(xnorsum10, vone);
              const auto cnt16_11 = _mm256_maddubs_epi16(xnorsum11, vone);
              const auto cnt16_20 = _mm256_maddubs_epi16(xnorsum20, vone);
              const auto cnt16_21 = _mm256_maddubs_epi16(xnorsum21, vone);
              const auto v11 = _mm256_set1_epi16(0x0001);
              const auto cnt32_00 = _mm256_madd_epi16(cnt16_00, v11);
              const auto cnt32_01 = _mm256_madd_epi16(cnt16_01, v11);
              const auto cnt32_10 = _mm256_madd_epi16(cnt16_10, v11);
              const auto cnt32_11 = _mm256_madd_epi16(cnt16_11, v11);
              const auto cnt32_20 = _mm256_madd_epi16(cnt16_20, v11);
              const auto cnt32_21 = _mm256_madd_epi16(cnt16_21, v11);
              const auto packed00 = _mm256_packs_epi32(cnt32_00, cnt32_01);
              const auto packed01 = _mm256_packs_epi32(cnt32_10, cnt32_11);
              const auto packed02 = _mm256_packs_epi32(cnt32_20, cnt32_21);
              const auto permed00 = _mm256_permute4x64_epi64(packed00, 0xD8);
              const auto permed01 = _mm256_permute4x64_epi64(packed01, 0xD8);
              const auto permed02 = _mm256_permute4x64_epi64(packed02, 0xD8);
              const auto v12 = _mm256_set1_epi32(0x00020001);
              const auto hlpacked0 = _mm256_madd_epi16(permed00, v12);
              const auto hlpacked1 = _mm256_madd_epi16(permed01, v12);
              const auto hlpacked2 = _mm256_madd_epi16(permed02, v12);
              const auto packed10 = _mm256_packs_epi32(hlpacked0, _mm256_setzero_si256());
              const auto packed11 = _mm256_packs_epi32(hlpacked1, _mm256_setzero_si256());
              const auto packed12 = _mm256_packs_epi32(hlpacked2, _mm256_setzero_si256());
              const auto permed10 = _mm256_permute4x64_epi64(packed10, 0xD8);
              const auto permed11 = _mm256_permute4x64_epi64(packed11, 0xD8);
              const auto permed12 = _mm256_permute4x64_epi64(packed12, 0xD8);
              const auto short0 = _mm256_castsi256_si128(permed10);
              const auto short1 = _mm256_castsi256_si128(permed11);
              const auto short2 = _mm256_castsi256_si128(permed12);
              const auto tmp0 = _mm_load_si128(reinterpret_cast<__m128i*>(&out_tile[row][col + 0][0]));
              const auto tmp1 = _mm_load_si128(reinterpret_cast<__m128i*>(&out_tile[row][col + 1][0]));
              const auto tmp2 = _mm_load_si128(reinterpret_cast<__m128i*>(&out_tile[row][col + 2][0]));
              const auto nsum = _mm_load_si128(reinterpret_cast<__m128i*>(&notsum[0]));
              const auto diff0 = _mm_sub_epi16(short0, nsum);
              const auto diff1 = _mm_sub_epi16(short1, nsum);
              const auto diff2 = _mm_sub_epi16(short2, nsum);
              const auto res0 = _mm_add_epi16(tmp0, diff0);
              const auto res1 = _mm_add_epi16(tmp1, diff1);
              const auto res2 = _mm_add_epi16(tmp2, diff2);
              _mm_store_si128(reinterpret_cast<__m128i*>(&out_tile[row][col + 0][0]), res0);
              _mm_store_si128(reinterpret_cast<__m128i*>(&out_tile[row][col + 1][0]), res1);
              _mm_store_si128(reinterpret_cast<__m128i*>(&out_tile[row][col + 2][0]), res2);
            }
          }
        }
      }
      if (p.thresholds != nullptr) {
        const auto th0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(p.thresholds + 0 * out_channels + out_ch_high * OutChUnroll));
        const auto th1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(p.thresholds + 1 * out_channels + out_ch_high * OutChUnroll));
        const auto th2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(p.thresholds + 2 * out_channels + out_ch_high * OutChUnroll));
        const auto flg = _mm_loadu_si128(reinterpret_cast<__m128i*>(p.thresholds + 3 * out_channels + out_ch_high * OutChUnroll));
        const auto is_neg = _mm_cmpgt_epi16(_mm_setzero_si128(), flg);
        const auto m2 = _mm_sub_epi16(flg, _mm_set1_epi16(2));
        const auto is_not_const = _mm_cmpgt_epi16(_mm_setzero_si128(), m2);
        for (std::size_t row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (std::size_t col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            const auto vec = _mm_loadu_si128(reinterpret_cast<__m128i*>(&out_tile[row][col][0]));
            const auto f0 = _mm_andnot_si128(_mm_cmpgt_epi16(th0, vec), flg);
            const auto f1 = _mm_andnot_si128(_mm_cmpgt_epi16(th1, vec), flg);
            const auto f2 = _mm_andnot_si128(_mm_cmpgt_epi16(th2, vec), flg);
            const auto tmp = _mm_add_epi16(_mm_add_epi16(f0, f1), _mm_add_epi16(f2, is_neg));
            const auto res = _mm_blendv_epi8(m2, tmp, is_not_const);
            const auto pres = _mm_packs_epi16(res, _mm_setzero_si128());
            const auto vlsb = _mm_slli_epi32(pres, 7);
            const auto vmsb = _mm_slli_epi32(pres, 6);
            const auto lsb = _mm_movemask_epi8(vlsb);
            const auto msb = _mm_movemask_epi8(vmsb);
            const auto Ohh = out_ch_high / OutChBlocks;
            const auto Om = out_ch_high % OutChBlocks;
            const auto index = Ohh * out_height * out_width * 2 * OutChBlocks
                + (row_high + row) * out_width * 2 * OutChBlocks
                + (col_high + col) * 2 * OutChBlocks
                + Om;
            reinterpret_cast<uint8_t*>(p.device_output_buf)[index + 0] = lsb;
            reinterpret_cast<uint8_t*>(p.device_output_buf)[index + OutChBlocks] = msb;
          }
        }
      } else {
        auto out_buf = reinterpret_cast<BIN_CONV_OUTPUT*>(p.device_output_buf);
        for (std::size_t row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (std::size_t col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            const auto vec = _mm_load_si128(reinterpret_cast<__m128i*>(&out_tile[row][col][0]));
            const auto Ohh = out_ch_high / OutChBlocks;
            const auto Om = out_ch_high % OutChBlocks;
            const auto index = Ohh * out_height * out_width * OutChUnroll2
                + (row_high + row) * out_width * OutChUnroll2
                + (col_high + col) * OutChUnroll2
                + Om * OutChUnroll;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_buf + index), vec);
          }
        }
      }
    }
  }
  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

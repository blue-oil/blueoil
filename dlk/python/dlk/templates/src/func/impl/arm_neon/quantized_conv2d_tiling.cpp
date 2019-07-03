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

#include "global.h"
#include "func/impl/apply_thresholds.h"
#include "func/impl/quantized_conv2d_tiling.h"
#include "func/impl/pack_16bit.h"
#include "time_measurement.h"
#include "tensor_convert.h"

#include <arm_neon.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dlk {

namespace impl {

void pack_input_for_tiling(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& input,
    const tiling_input_t& output) {
  Measurement::Start("Pack_input_for_tiling");
  const T_UINT in_channels = input.get_shape()[3];
  const T_UINT in_height = input.get_shape()[1];
  const T_UINT in_width = input.get_shape()[2];
  const T_UINT in_bitwidth = output.get_shape()[3];
  
  constexpr T_UINT InTypeBitWidth = CHAR_BIT * sizeof(uint32_t);
  const T_UINT in_stride = (in_channels + InTypeBitWidth - 1) / InTypeBitWidth;
#pragma omp parallel for schedule(dynamic)
  for (unsigned int in_ch_high = 0; in_ch_high < in_stride; ++in_ch_high) {
    for (unsigned int row = 0; row < in_height; ++row) {
      for (unsigned int col = 0; col < in_width; ++col) {
        for (unsigned int in_bit_ch = 0; in_bit_ch < in_bitwidth; ++in_bit_ch) {
          output(in_ch_high, row, col, in_bit_ch, 0) = tiling_input_elem_t(0);
        }
      }
    }
  }
#pragma omp parallel for schedule(dynamic)
  for (unsigned int row = 0; row < in_height; ++row) {
    for (unsigned int col = 0; col < in_width; ++col) {
      for (unsigned int in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
        for (unsigned int in_ch_low = 0; in_ch_low < InTypeBitWidth; ++in_ch_low) {
          unsigned int in_ch = in_ch_high + in_ch_low;
          if (in_ch >= in_channels) break;
          QUANTIZED_NOT_PACKED val = input(0, row, col, in_ch);
          for (unsigned int in_bit_ch = 0; in_bit_ch < in_bitwidth; ++in_bit_ch) {
            tiling_input_elem_base_t bit = (val >> in_bit_ch) & 1;
            output(in_ch_high / InTypeBitWidth, row, col, in_bit_ch, 0) |= tiling_input_elem_t(bit << in_ch_low);
          }
        }
      }
    }
  }

  Measurement::Stop();
}

void QuantizedConv2DTiling(const tiling_input_t& input,
                                  const kernel_t& kernel,
                                  const binary_convolution_parameters &p) {
  constexpr T_UINT InTypeBitWidth = tiling_input_elem_t::BitCount;
  convolution_parameters cp = p.normal_conv_params;
  const T_UINT out_channels = cp.output_channels;
  const T_UINT kh = cp.kernel_height;
  const T_UINT kw = cp.kernel_width;
  const T_UINT in_bitwidth = 2;
  const T_UINT in_channels = cp.kernel_depth;
  const T_UINT in_height = cp.input_height;
  const T_UINT in_width = cp.input_width;
  const T_UINT in_stride = (in_channels + InTypeBitWidth - 1) / InTypeBitWidth;
  const T_UINT padding = cp.padding;
  const T_UINT out_height = cp.output_height;
  const T_UINT out_width = cp.output_width;
  const T_UINT out_size = out_height * out_width * out_channels;

  assert(kh * kw < 32);
  assert(in_height * in_width == out_height * out_width);
  assert((in_channels % InTypeBitWidth) == 0);

#ifdef AARCH32
  const T_UINT TileHeight = std::min(in_height, T_UINT(32)); // configurable
  const T_UINT TileWidth = std::min(in_width, T_UINT(32)); // configurable
  constexpr T_UINT InChUnroll = InTypeBitWidth; // hardcoded, not configurable
  constexpr T_UINT OutChUnroll = 16; // hardcoded, not configurable
  constexpr T_UINT InBitChUnroll = 2; // hardcoded, not configurable

  const T_UINT row_tile_count = (in_height + TileHeight - 1) / TileHeight;
  const T_UINT col_tile_count = (in_width + TileWidth - 1) / TileWidth;
  const T_UINT out_tile_count = (out_channels + OutChUnroll - 1) / OutChUnroll;
  const T_UINT total_tile_count = row_tile_count * col_tile_count * out_tile_count;
  Measurement::Start("Quantized Conv2D Tiling");
#pragma omp parallel for
  for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
    T_UINT out_ch_high = tile_index % out_tile_count * OutChUnroll;
    T_UINT col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
    T_UINT row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
    int16_t out_tile[TileHeight][TileWidth][OutChUnroll];
    for (unsigned int row = 0; row < TileHeight; ++row) {
      for (unsigned int col = 0; col < TileWidth; ++col) {
        for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          out_tile[row][col][out_ch] = 0;
        }
      }
    }
    for (unsigned int in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
      QUANTIZED_PACKED_KERNEL notk[kh][kw][OutChUnroll];
      int16_t notsum[OutChUnroll] = {};
      for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
        notsum[out_ch] = 0;
        for (unsigned int kr = 0; kr < kh; ++kr) {
          for (unsigned int kc = 0; kc < kw; ++kc) {
            notk[kr][kc][out_ch] = kernel(out_ch_high + out_ch, kr, kc, in_ch_high / InTypeBitWidth);
            notsum[out_ch] += pop_count(notk[kr][kc][out_ch]);
          }
        }
      }
      for (unsigned int in_bit_ch_high = 0; in_bit_ch_high < in_bitwidth; in_bit_ch_high += InBitChUnroll) {
        tiling_input_elem_t in_tile[TileHeight + kh - 1][TileWidth + kw - 1][InBitChUnroll];
        for (unsigned int row = 0; row < TileHeight + kh - 1; ++row) {
          if (row_high + row >= in_height + 2*padding) break;
          for (unsigned int col = 0; col < TileWidth + kw - 1; ++col) {
            if (col_high + col >= in_width + 2*padding) break;
            for (unsigned int in_bit_ch = 0; in_bit_ch < InBitChUnroll; ++in_bit_ch) {
              if (row_high + row < padding || row_high + row >= in_height + padding
                  || col_high + col < padding || col_high + col >= in_width + padding) {
                in_tile[row][col][in_bit_ch] = tiling_input_elem_t(0);
              } else {
                in_tile[row][col][in_bit_ch] = input(in_ch_high / InTypeBitWidth, row_high + row - padding,
                    col_high + col - padding, in_bit_ch_high + in_bit_ch, 0);
              }
            }
          }
        }
        for (unsigned int row = 0; row < TileHeight; ++row) {
          for (unsigned int col = 0; col < TileWidth; ++col) {
            uint8x16_t xnorsum00 = vdupq_n_u8(0);
            uint8x16_t xnorsum01 = vdupq_n_u8(0);
            uint8x16_t xnorsum10 = vdupq_n_u8(0);
            uint8x16_t xnorsum11 = vdupq_n_u8(0);
            uint8x16_t xnorsum20 = vdupq_n_u8(0);
            uint8x16_t xnorsum21 = vdupq_n_u8(0);
            uint8x16_t xnorsum30 = vdupq_n_u8(0);
            uint8x16_t xnorsum31 = vdupq_n_u8(0);
            for (unsigned int kr = 0; kr < kh; ++kr) {
              for (unsigned int kc = 0; kc < kw; ++kc) {
                uint32x4_t nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 0]));
                uint32x4_t nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 4]));
                uint32x4_t nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 8]));
                uint32x4_t nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][12]));
                uint8x16_t nk08 = vreinterpretq_u8_u32(nk0);
                uint8x16_t nk18 = vreinterpretq_u8_u32(nk1);
                uint8x16_t nk28 = vreinterpretq_u8_u32(nk2);
                uint8x16_t nk38 = vreinterpretq_u8_u32(nk3);
                uint32x4_t in = vdupq_n_u32(in_tile[row + kr][col + kc][0].Raw());
                uint8x16_t in8 = vreinterpretq_u8_u32(in);
                xnorsum00 += vcntq_u8(in8 ^ nk08);
                xnorsum10 += vcntq_u8(in8 ^ nk18);
                xnorsum20 += vcntq_u8(in8 ^ nk28);
                xnorsum30 += vcntq_u8(in8 ^ nk38);
                in = vdupq_n_u32(in_tile[row + kr][col + kc][1].Raw());
                in8 = vreinterpretq_u8_u32(in);
                xnorsum01 += vcntq_u8(in8 ^ nk08);
                xnorsum11 += vcntq_u8(in8 ^ nk18);
                xnorsum21 += vcntq_u8(in8 ^ nk28);
                xnorsum31 += vcntq_u8(in8 ^ nk38);
              }
            }
            uint16x8_t psum000 = vpaddlq_u8(xnorsum00);
            uint16x8_t psum010 = vpaddlq_u8(xnorsum10);
            uint16x8_t psum020 = vpaddlq_u8(xnorsum20);
            uint16x8_t psum030 = vpaddlq_u8(xnorsum30);
            uint16x8_t psum001 = vpaddlq_u8(xnorsum01);
            uint16x8_t psum011 = vpaddlq_u8(xnorsum11);
            uint16x8_t psum021 = vpaddlq_u8(xnorsum21);
            uint16x8_t psum031 = vpaddlq_u8(xnorsum31);
            uint32x4_t psum100 = vpaddlq_u16(psum000);
            uint32x4_t psum110 = vpaddlq_u16(psum010);
            uint32x4_t psum120 = vpaddlq_u16(psum020);
            uint32x4_t psum130 = vpaddlq_u16(psum030);
            uint32x4_t psum101 = vpaddlq_u16(psum001);
            uint32x4_t psum111 = vpaddlq_u16(psum011);
            uint32x4_t psum121 = vpaddlq_u16(psum021);
            uint32x4_t psum131 = vpaddlq_u16(psum031);
            uint16x8_t usum010 = vcombine_u16(vmovn_u32(psum100), vmovn_u32(psum110));
            uint16x8_t usum230 = vcombine_u16(vmovn_u32(psum120), vmovn_u32(psum130));
            uint16x8_t usum011 = vcombine_u16(vmovn_u32(psum101), vmovn_u32(psum111));
            uint16x8_t usum231 = vcombine_u16(vmovn_u32(psum121), vmovn_u32(psum131));
            int16x8_t sum010 = vreinterpretq_s16_u16(usum010);
            int16x8_t sum230 = vreinterpretq_s16_u16(usum230);
            int16x8_t sum011 = vreinterpretq_s16_u16(usum011);
            int16x8_t sum231 = vreinterpretq_s16_u16(usum231);
            int16x8_t tmp0 = vld1q_s16(&out_tile[row][col][0]);
            int16x8_t tmp1 = vld1q_s16(&out_tile[row][col][8]);
            int16x8_t nsum0 = vld1q_s16(&notsum[0]);
            int16x8_t nsum1 = vld1q_s16(&notsum[8]);
            tmp0 += vshlq_s16(sum010 - nsum0, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum011 - nsum0, vdupq_n_s16(in_bit_ch_high + 1));
            tmp1 += vshlq_s16(sum230 - nsum1, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum231 - nsum1, vdupq_n_s16(in_bit_ch_high + 1));
            vst1q_s16(&out_tile[row][col][0], tmp0);
            vst1q_s16(&out_tile[row][col][8], tmp1);
          }
        }
      }
    }
    for (unsigned int row = 0; row < TileHeight; ++row) {
      if (row_high + row >= out_height) break;
      for (unsigned int col = 0; col < TileWidth; ++col) {
        if (col_high + col >= out_width) break;
        for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          unsigned int index = (row_high + row) * out_width * out_channels
              + (col_high + col) * out_channels
              + (out_ch_high + out_ch);
          p.device_output_buf[index] = out_tile[row][col][out_ch];
        }
      }
    }
  }
#else
  const T_UINT TileHeight = std::min(in_height, T_UINT(32)); // configurable
  const T_UINT TileWidth = std::min(in_width, T_UINT(32)); // configurable
  constexpr T_UINT InChUnroll = InTypeBitWidth; // hardcoded, not configurable
  constexpr T_UINT OutChUnroll = 32; // hardcoded, not configurable
  constexpr T_UINT InBitChUnroll = 2; // hardcoded, not configurable

  const T_UINT row_tile_count = (in_height + TileHeight - 1) / TileHeight;
  const T_UINT col_tile_count = (in_width + TileWidth - 1) / TileWidth;
  const T_UINT out_tile_count = (out_channels + OutChUnroll - 1) / OutChUnroll;
  const T_UINT total_tile_count = row_tile_count * col_tile_count * out_tile_count;
  Measurement::Start("Quantized Conv2D Tiling");
#pragma omp parallel for
  for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
    T_UINT out_ch_high = tile_index % out_tile_count * OutChUnroll;
    T_UINT col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
    T_UINT row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
    int16_t out_tile[TileHeight][TileWidth][OutChUnroll];
    for (unsigned int row = 0; row < TileHeight; ++row) {
      for (unsigned int col = 0; col < TileWidth; ++col) {
        for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          out_tile[row][col][out_ch] = 0;
        }
      }
    }
    for (unsigned int in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
      QUANTIZED_PACKED_KERNEL notk[kh][kw][OutChUnroll];
      int16_t notsum[OutChUnroll] = {};
      for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
        notsum[out_ch] = 0;
        for (unsigned int kr = 0; kr < kh; ++kr) {
          for (unsigned int kc = 0; kc < kw; ++kc) {
            notk[kr][kc][out_ch] = kernel(out_ch_high + out_ch, kr, kc, in_ch_high / InTypeBitWidth);
            notsum[out_ch] += pop_count(notk[kr][kc][out_ch]);
          }
        }
      }
      for (unsigned int in_bit_ch_high = 0; in_bit_ch_high < in_bitwidth; in_bit_ch_high += InBitChUnroll) {
        tiling_input_elem_t in_tile[TileHeight + kh - 1][TileWidth + kw - 1][InBitChUnroll];
        for (unsigned int row = 0; row < TileHeight + kh - 1; ++row) {
          if (row_high + row >= in_height + 2*padding) break;
          for (unsigned int col = 0; col < TileWidth + kw - 1; ++col) {
            if (col_high + col >= in_width + 2*padding) break;
            for (unsigned int in_bit_ch = 0; in_bit_ch < InBitChUnroll; ++in_bit_ch) {
              if (row_high + row < padding || row_high + row >= in_height + padding
                  || col_high + col < padding || col_high + col >= in_width + padding) {
                in_tile[row][col][in_bit_ch] = tiling_input_elem_t(0);
              } else {
                in_tile[row][col][in_bit_ch] = input(in_ch_high / InTypeBitWidth, row_high + row - padding,
                    col_high + col - padding, in_bit_ch_high + in_bit_ch, 0);
              }
            }
          }
        }
        for (unsigned int row = 0; row < TileHeight; ++row) {
          for (unsigned int col = 0; col < TileWidth; ++col) {
            uint8x16_t xnorsum00 = vdupq_n_u8(0);
            uint8x16_t xnorsum01 = vdupq_n_u8(0);
            uint8x16_t xnorsum10 = vdupq_n_u8(0);
            uint8x16_t xnorsum11 = vdupq_n_u8(0);
            uint8x16_t xnorsum20 = vdupq_n_u8(0);
            uint8x16_t xnorsum21 = vdupq_n_u8(0);
            uint8x16_t xnorsum30 = vdupq_n_u8(0);
            uint8x16_t xnorsum31 = vdupq_n_u8(0);
            uint8x16_t xnorsum40 = vdupq_n_u8(0);
            uint8x16_t xnorsum41 = vdupq_n_u8(0);
            uint8x16_t xnorsum50 = vdupq_n_u8(0);
            uint8x16_t xnorsum51 = vdupq_n_u8(0);
            uint8x16_t xnorsum60 = vdupq_n_u8(0);
            uint8x16_t xnorsum61 = vdupq_n_u8(0);
            uint8x16_t xnorsum70 = vdupq_n_u8(0);
            uint8x16_t xnorsum71 = vdupq_n_u8(0);
            for (unsigned int kr = 0; kr < kh; ++kr) {
              for (unsigned int kc = 0; kc < kw; ++kc) {
                uint32x4_t nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 0]));
                uint32x4_t nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 4]));
                uint32x4_t nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 8]));
                uint32x4_t nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][12]));
                uint32x4_t nk4 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][16]));
                uint32x4_t nk5 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][20]));
                uint32x4_t nk6 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][24]));
                uint32x4_t nk7 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][28]));
                uint8x16_t nk08 = vreinterpretq_u8_u32(nk0);
                uint8x16_t nk18 = vreinterpretq_u8_u32(nk1);
                uint8x16_t nk28 = vreinterpretq_u8_u32(nk2);
                uint8x16_t nk38 = vreinterpretq_u8_u32(nk3);
                uint8x16_t nk48 = vreinterpretq_u8_u32(nk4);
                uint8x16_t nk58 = vreinterpretq_u8_u32(nk5);
                uint8x16_t nk68 = vreinterpretq_u8_u32(nk6);
                uint8x16_t nk78 = vreinterpretq_u8_u32(nk7);
                uint32x4_t in = vdupq_n_u32(in_tile[row + kr][col + kc][0].Raw());
                uint8x16_t in8 = vreinterpretq_u8_u32(in);
                xnorsum00 += vcntq_u8(in8 ^ nk08);
                xnorsum10 += vcntq_u8(in8 ^ nk18);
                xnorsum20 += vcntq_u8(in8 ^ nk28);
                xnorsum30 += vcntq_u8(in8 ^ nk38);
                xnorsum40 += vcntq_u8(in8 ^ nk48);
                xnorsum50 += vcntq_u8(in8 ^ nk58);
                xnorsum60 += vcntq_u8(in8 ^ nk68);
                xnorsum70 += vcntq_u8(in8 ^ nk78);
                in = vdupq_n_u32(in_tile[row + kr][col + kc][1].Raw());
                in8 = vreinterpretq_u8_u32(in);
                xnorsum01 += vcntq_u8(in8 ^ nk08);
                xnorsum11 += vcntq_u8(in8 ^ nk18);
                xnorsum21 += vcntq_u8(in8 ^ nk28);
                xnorsum31 += vcntq_u8(in8 ^ nk38);
                xnorsum41 += vcntq_u8(in8 ^ nk48);
                xnorsum51 += vcntq_u8(in8 ^ nk58);
                xnorsum61 += vcntq_u8(in8 ^ nk68);
                xnorsum71 += vcntq_u8(in8 ^ nk78);
              }
            }
            uint16x8_t psum000 = vpaddlq_u8(xnorsum00);
            uint16x8_t psum010 = vpaddlq_u8(xnorsum10);
            uint16x8_t psum020 = vpaddlq_u8(xnorsum20);
            uint16x8_t psum030 = vpaddlq_u8(xnorsum30);
            uint16x8_t psum040 = vpaddlq_u8(xnorsum40);
            uint16x8_t psum050 = vpaddlq_u8(xnorsum50);
            uint16x8_t psum060 = vpaddlq_u8(xnorsum60);
            uint16x8_t psum070 = vpaddlq_u8(xnorsum70);
            uint16x8_t psum001 = vpaddlq_u8(xnorsum01);
            uint16x8_t psum011 = vpaddlq_u8(xnorsum11);
            uint16x8_t psum021 = vpaddlq_u8(xnorsum21);
            uint16x8_t psum031 = vpaddlq_u8(xnorsum31);
            uint16x8_t psum041 = vpaddlq_u8(xnorsum41);
            uint16x8_t psum051 = vpaddlq_u8(xnorsum51);
            uint16x8_t psum061 = vpaddlq_u8(xnorsum61);
            uint16x8_t psum071 = vpaddlq_u8(xnorsum71);
            uint32x4_t psum100 = vpaddlq_u16(psum000);
            uint32x4_t psum110 = vpaddlq_u16(psum010);
            uint32x4_t psum120 = vpaddlq_u16(psum020);
            uint32x4_t psum130 = vpaddlq_u16(psum030);
            uint32x4_t psum140 = vpaddlq_u16(psum040);
            uint32x4_t psum150 = vpaddlq_u16(psum050);
            uint32x4_t psum160 = vpaddlq_u16(psum060);
            uint32x4_t psum170 = vpaddlq_u16(psum070);
            uint32x4_t psum101 = vpaddlq_u16(psum001);
            uint32x4_t psum111 = vpaddlq_u16(psum011);
            uint32x4_t psum121 = vpaddlq_u16(psum021);
            uint32x4_t psum131 = vpaddlq_u16(psum031);
            uint32x4_t psum141 = vpaddlq_u16(psum041);
            uint32x4_t psum151 = vpaddlq_u16(psum051);
            uint32x4_t psum161 = vpaddlq_u16(psum061);
            uint32x4_t psum171 = vpaddlq_u16(psum071);
            uint16x8_t usum010 = vcombine_u16(vmovn_u32(psum100), vmovn_u32(psum110));
            uint16x8_t usum230 = vcombine_u16(vmovn_u32(psum120), vmovn_u32(psum130));
            uint16x8_t usum450 = vcombine_u16(vmovn_u32(psum140), vmovn_u32(psum150));
            uint16x8_t usum670 = vcombine_u16(vmovn_u32(psum160), vmovn_u32(psum170));
            uint16x8_t usum011 = vcombine_u16(vmovn_u32(psum101), vmovn_u32(psum111));
            uint16x8_t usum231 = vcombine_u16(vmovn_u32(psum121), vmovn_u32(psum131));
            uint16x8_t usum451 = vcombine_u16(vmovn_u32(psum141), vmovn_u32(psum151));
            uint16x8_t usum671 = vcombine_u16(vmovn_u32(psum161), vmovn_u32(psum171));
            int16x8_t sum010 = vreinterpretq_s16_u16(usum010);
            int16x8_t sum230 = vreinterpretq_s16_u16(usum230);
            int16x8_t sum450 = vreinterpretq_s16_u16(usum450);
            int16x8_t sum670 = vreinterpretq_s16_u16(usum670);
            int16x8_t sum011 = vreinterpretq_s16_u16(usum011);
            int16x8_t sum231 = vreinterpretq_s16_u16(usum231);
            int16x8_t sum451 = vreinterpretq_s16_u16(usum451);
            int16x8_t sum671 = vreinterpretq_s16_u16(usum671);
            int16x8_t tmp0 = vld1q_s16(&out_tile[row][col][ 0]);
            int16x8_t tmp1 = vld1q_s16(&out_tile[row][col][ 8]);
            int16x8_t tmp2 = vld1q_s16(&out_tile[row][col][16]);
            int16x8_t tmp3 = vld1q_s16(&out_tile[row][col][24]);
            int16x8_t nsum0 = vld1q_s16(&notsum[ 0]);
            int16x8_t nsum1 = vld1q_s16(&notsum[ 8]);
            int16x8_t nsum2 = vld1q_s16(&notsum[16]);
            int16x8_t nsum3 = vld1q_s16(&notsum[24]);
            tmp0 += vshlq_s16(sum010 - nsum0, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum011 - nsum0, vdupq_n_s16(in_bit_ch_high + 1));
            tmp1 += vshlq_s16(sum230 - nsum1, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum231 - nsum1, vdupq_n_s16(in_bit_ch_high + 1));
            tmp2 += vshlq_s16(sum450 - nsum2, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum451 - nsum2, vdupq_n_s16(in_bit_ch_high + 1));
            tmp3 += vshlq_s16(sum670 - nsum3, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum671 - nsum3, vdupq_n_s16(in_bit_ch_high + 1));
            vst1q_s16(&out_tile[row][col][ 0], tmp0);
            vst1q_s16(&out_tile[row][col][ 8], tmp1);
            vst1q_s16(&out_tile[row][col][16], tmp2);
            vst1q_s16(&out_tile[row][col][24], tmp3);
          }
        }
      }
    }
    for (unsigned int row = 0; row < TileHeight; ++row) {
      if (row_high + row >= out_height) break;
      for (unsigned int col = 0; col < TileWidth; ++col) {
        if (col_high + col >= out_width) break;
        for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          unsigned int index = (row_high + row) * out_width * out_channels
              + (col_high + col) * out_channels
              + (out_ch_high + out_ch);
          p.device_output_buf[index] = out_tile[row][col][out_ch];
        }
      }
    }
  }
#endif
  Measurement::Stop();

  using namespace dlk;
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, out_channels, in_height * in_width);

  if (p.thresholds != nullptr) {
    ApplyThresholds(output_, p);
    const auto buf = std::make_unique<QUANTIZED_PACKED[]>(out_size * p.n_bit / CHAR_BIT);
    pack_16bit(p.device_output_buf, buf.get(), out_size);
    const std::size_t b = 32;
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>::tensor_info_t<std::size_t> buf_shape = {
      out_height, out_width, (out_channels + b - 1) / b, p.n_bit, b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl> buf_tensor(buf.get(), buf_shape);
    TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>::tensor_info_t<std::size_t> out_shape = {
      (out_channels + b - 1) / b,
      out_height,
      out_width,
      p.n_bit,
      b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl> out((QUANTIZED_PACKED*)p.device_output_buf, out_shape);
    convert_tensor(buf_tensor, out);
  } else {
    const std::size_t b = 32;
    const auto buf = std::make_unique<BIN_CONV_OUTPUT[]>(out_size);
    std::copy(p.device_output_buf, p.device_output_buf + out_size, buf.get());
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>::tensor_info_t<std::size_t> buf_shape = {
      out_height, out_width, out_channels
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC> buf_tensor(buf.get(), buf_shape);
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl>::tensor_info_t<std::size_t> out_shape = {
      (out_channels + b - 1) / b, out_height, out_width, b
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl> out(p.device_output_buf, out_shape);
    convert_tensor(buf_tensor, out);
  }
}

} // namespace impl

} // namespace dlk

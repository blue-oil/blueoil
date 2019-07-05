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
            const auto index = (out_ch_high + out_ch) * kh * kw * (in_channels / InTypeBitWidth)
              + kr * kw * (in_channels / InTypeBitWidth)
              + kc * (in_channels / InTypeBitWidth)
              + in_ch_high / InTypeBitWidth;
            notk[kr][kc][out_ch] = kernel.data()[index];
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
                const auto index = (in_ch_high / InTypeBitWidth) * in_height * in_width * in_bitwidth
                  + (row_high + row - padding) * in_width * in_bitwidth
                  + (col_high + col - padding) * in_bitwidth
                  + (in_bit_ch_high + in_bit_ch);
                in_tile[row][col][in_bit_ch] = input.data()[index];
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
  const T_UINT TileWidth = std::min(in_width + (in_width & 1), T_UINT(32)); // configurable
  constexpr T_UINT InChUnroll = InTypeBitWidth; // hardcoded, not configurable
  constexpr T_UINT OutChUnroll = 16; // hardcoded, not configurable
  constexpr T_UINT InBitChUnroll = 2; // hardcoded, not configurable
  constexpr T_UINT ColUnroll = 2; // hardcoded, not configurable

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
            const auto index = (out_ch_high + out_ch) * kh * kw * (in_channels / InTypeBitWidth)
              + kr * kw * (in_channels / InTypeBitWidth)
              + kc * (in_channels / InTypeBitWidth)
              + in_ch_high / InTypeBitWidth;
            notk[kr][kc][out_ch] = kernel.data()[index];
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
                const auto index = (in_ch_high / InTypeBitWidth) * in_height * in_width * in_bitwidth
                  + (row_high + row - padding) * in_width * in_bitwidth
                  + (col_high + col - padding) * in_bitwidth
                  + (in_bit_ch_high + in_bit_ch);
                in_tile[row][col][in_bit_ch] = input.data()[index];
              }
            }
          }
        }
        for (unsigned int row = 0; row < TileHeight; ++row) {
          for (unsigned int col = 0; col < TileWidth; col += ColUnroll) {
            uint8x16_t xnorsum000 = vdupq_n_u8(0);
            uint8x16_t xnorsum001 = vdupq_n_u8(0);
            uint8x16_t xnorsum010 = vdupq_n_u8(0);
            uint8x16_t xnorsum011 = vdupq_n_u8(0);
            uint8x16_t xnorsum020 = vdupq_n_u8(0);
            uint8x16_t xnorsum021 = vdupq_n_u8(0);
            uint8x16_t xnorsum030 = vdupq_n_u8(0);
            uint8x16_t xnorsum031 = vdupq_n_u8(0);
            uint8x16_t xnorsum100 = vdupq_n_u8(0);
            uint8x16_t xnorsum101 = vdupq_n_u8(0);
            uint8x16_t xnorsum110 = vdupq_n_u8(0);
            uint8x16_t xnorsum111 = vdupq_n_u8(0);
            uint8x16_t xnorsum120 = vdupq_n_u8(0);
            uint8x16_t xnorsum121 = vdupq_n_u8(0);
            uint8x16_t xnorsum130 = vdupq_n_u8(0);
            uint8x16_t xnorsum131 = vdupq_n_u8(0);
            for (unsigned int kr = 0; kr < kh; ++kr) {
              uint32x4_t inl0 = vdupq_n_u32(in_tile[row + kr][col][0].Raw());
              uint32x4_t inh0 = vdupq_n_u32(in_tile[row + kr][col][1].Raw());
              uint8x16_t inl08 = vreinterpretq_u8_u32(inl0);
              uint8x16_t inh08 = vreinterpretq_u8_u32(inh0);
              for (unsigned int kc = 0; kc < kw; ++kc) {
                uint32x4_t nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 0]));
                uint32x4_t nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 4]));
                uint32x4_t nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][ 8]));
                uint32x4_t nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[kr][kc][12]));
                uint8x16_t nk08 = vreinterpretq_u8_u32(nk0);
                uint8x16_t nk18 = vreinterpretq_u8_u32(nk1);
                uint8x16_t nk28 = vreinterpretq_u8_u32(nk2);
                uint8x16_t nk38 = vreinterpretq_u8_u32(nk3);
                uint32x4_t inl1 = vdupq_n_u32(in_tile[row + kr][col + kc + 1][0].Raw());
                uint8x16_t inl18 = vreinterpretq_u8_u32(inl1);
                xnorsum000 += vcntq_u8(inl08 ^ nk08);
                xnorsum010 += vcntq_u8(inl08 ^ nk18);
                xnorsum020 += vcntq_u8(inl08 ^ nk28);
                xnorsum030 += vcntq_u8(inl08 ^ nk38);
                xnorsum100 += vcntq_u8(inl18 ^ nk08);
                xnorsum110 += vcntq_u8(inl18 ^ nk18);
                xnorsum120 += vcntq_u8(inl18 ^ nk28);
                xnorsum130 += vcntq_u8(inl18 ^ nk38);
                inl08 = inl18;
                uint32x4_t inh1 = vdupq_n_u32(in_tile[row + kr][col + kc + 1][1].Raw());
                uint8x16_t inh18 = vreinterpretq_u8_u32(inh1);
                xnorsum001 += vcntq_u8(inh08 ^ nk08);
                xnorsum011 += vcntq_u8(inh08 ^ nk18);
                xnorsum021 += vcntq_u8(inh08 ^ nk28);
                xnorsum031 += vcntq_u8(inh08 ^ nk38);
                xnorsum101 += vcntq_u8(inh18 ^ nk08);
                xnorsum111 += vcntq_u8(inh18 ^ nk18);
                xnorsum121 += vcntq_u8(inh18 ^ nk28);
                xnorsum131 += vcntq_u8(inh18 ^ nk38);
                inh08 = inh18;
              }
            }
            uint16x8_t psum0000 = vpaddlq_u8(xnorsum000);
            uint16x8_t psum0010 = vpaddlq_u8(xnorsum010);
            uint16x8_t psum0020 = vpaddlq_u8(xnorsum020);
            uint16x8_t psum0030 = vpaddlq_u8(xnorsum030);
            uint16x8_t psum0001 = vpaddlq_u8(xnorsum001);
            uint16x8_t psum0011 = vpaddlq_u8(xnorsum011);
            uint16x8_t psum0021 = vpaddlq_u8(xnorsum021);
            uint16x8_t psum0031 = vpaddlq_u8(xnorsum031);
            uint16x8_t psum0100 = vpaddlq_u8(xnorsum100);
            uint16x8_t psum0110 = vpaddlq_u8(xnorsum110);
            uint16x8_t psum0120 = vpaddlq_u8(xnorsum120);
            uint16x8_t psum0130 = vpaddlq_u8(xnorsum130);
            uint16x8_t psum0101 = vpaddlq_u8(xnorsum101);
            uint16x8_t psum0111 = vpaddlq_u8(xnorsum111);
            uint16x8_t psum0121 = vpaddlq_u8(xnorsum121);
            uint16x8_t psum0131 = vpaddlq_u8(xnorsum131);
            uint16x8_t psum1000 = vreinterpretq_u16_u32(vpaddlq_u16(psum0000));
            uint16x8_t psum1010 = vreinterpretq_u16_u32(vpaddlq_u16(psum0010));
            uint16x8_t psum1020 = vreinterpretq_u16_u32(vpaddlq_u16(psum0020));
            uint16x8_t psum1030 = vreinterpretq_u16_u32(vpaddlq_u16(psum0030));
            uint16x8_t psum1001 = vreinterpretq_u16_u32(vpaddlq_u16(psum0001));
            uint16x8_t psum1011 = vreinterpretq_u16_u32(vpaddlq_u16(psum0011));
            uint16x8_t psum1021 = vreinterpretq_u16_u32(vpaddlq_u16(psum0021));
            uint16x8_t psum1031 = vreinterpretq_u16_u32(vpaddlq_u16(psum0031));
            uint16x8_t psum1100 = vreinterpretq_u16_u32(vpaddlq_u16(psum0100));
            uint16x8_t psum1110 = vreinterpretq_u16_u32(vpaddlq_u16(psum0110));
            uint16x8_t psum1120 = vreinterpretq_u16_u32(vpaddlq_u16(psum0120));
            uint16x8_t psum1130 = vreinterpretq_u16_u32(vpaddlq_u16(psum0130));
            uint16x8_t psum1101 = vreinterpretq_u16_u32(vpaddlq_u16(psum0101));
            uint16x8_t psum1111 = vreinterpretq_u16_u32(vpaddlq_u16(psum0111));
            uint16x8_t psum1121 = vreinterpretq_u16_u32(vpaddlq_u16(psum0121));
            uint16x8_t psum1131 = vreinterpretq_u16_u32(vpaddlq_u16(psum0131));
            uint16x8_t usum0010 = vuzpq_u16(psum1000, psum1010).val[0];
            uint16x8_t usum0230 = vuzpq_u16(psum1020, psum1030).val[0];
            uint16x8_t usum0011 = vuzpq_u16(psum1001, psum1011).val[0];
            uint16x8_t usum0231 = vuzpq_u16(psum1021, psum1031).val[0];
            uint16x8_t usum1010 = vuzpq_u16(psum1100, psum1110).val[0];
            uint16x8_t usum1230 = vuzpq_u16(psum1120, psum1130).val[0];
            uint16x8_t usum1011 = vuzpq_u16(psum1101, psum1111).val[0];
            uint16x8_t usum1231 = vuzpq_u16(psum1121, psum1131).val[0];
            int16x8_t sum0010 = vreinterpretq_s16_u16(usum0010);
            int16x8_t sum0230 = vreinterpretq_s16_u16(usum0230);
            int16x8_t sum0011 = vreinterpretq_s16_u16(usum0011);
            int16x8_t sum0231 = vreinterpretq_s16_u16(usum0231);
            int16x8_t sum1010 = vreinterpretq_s16_u16(usum1010);
            int16x8_t sum1230 = vreinterpretq_s16_u16(usum1230);
            int16x8_t sum1011 = vreinterpretq_s16_u16(usum1011);
            int16x8_t sum1231 = vreinterpretq_s16_u16(usum1231);
            int16x8_t tmp00 = vld1q_s16(&out_tile[row][col + 0][ 0]);
            int16x8_t tmp01 = vld1q_s16(&out_tile[row][col + 0][ 8]);
            int16x8_t tmp10 = vld1q_s16(&out_tile[row][col + 1][ 0]);
            int16x8_t tmp11 = vld1q_s16(&out_tile[row][col + 1][ 8]);
            int16x8_t nsum0 = vld1q_s16(&notsum[ 0]);
            int16x8_t nsum1 = vld1q_s16(&notsum[ 8]);
            tmp00 += vshlq_s16(sum0010 - nsum0, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum0011 - nsum0, vdupq_n_s16(in_bit_ch_high + 1));
            tmp01 += vshlq_s16(sum0230 - nsum1, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum0231 - nsum1, vdupq_n_s16(in_bit_ch_high + 1));
            tmp10 += vshlq_s16(sum1010 - nsum0, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum1011 - nsum0, vdupq_n_s16(in_bit_ch_high + 1));
            tmp11 += vshlq_s16(sum1230 - nsum1, vdupq_n_s16(in_bit_ch_high))
              + vshlq_s16(sum1231 - nsum1, vdupq_n_s16(in_bit_ch_high + 1));
            vst1q_s16(&out_tile[row][col + 0][ 0], tmp00);
            vst1q_s16(&out_tile[row][col + 0][ 8], tmp01);
            vst1q_s16(&out_tile[row][col + 1][ 0], tmp10);
            vst1q_s16(&out_tile[row][col + 1][ 8], tmp11);
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
    Measurement::Start("Output tensor convert");
    convert_tensor(buf_tensor, out);
    Measurement::Stop();
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
    Measurement::Start("Output tensor convert");
    convert_tensor(buf_tensor, out);
    Measurement::Stop();
  }
}

} // namespace impl

} // namespace dlk

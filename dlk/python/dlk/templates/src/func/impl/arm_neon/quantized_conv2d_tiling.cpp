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

static auto buf_th = std::make_unique<BIN_CONV_OUTPUT[]>(NUM_OF_A2W1_THRESHOLD * MAX_IN_C);

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

  if (p.thresholds != nullptr) {
    for (unsigned int i = 0; i < out_channels; ++i) {
      auto th0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 0];
      auto th1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
      auto th2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
      const auto flg = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
      if (flg == -1) {
        ++th0;
        ++th1;
        ++th2;
      }
      buf_th[NUM_OF_A2W1_THRESHOLD * i + 0] = th0;
      buf_th[NUM_OF_A2W1_THRESHOLD * i + 1] = th1;
      buf_th[NUM_OF_A2W1_THRESHOLD * i + 2] = th2;
      buf_th[NUM_OF_A2W1_THRESHOLD * i + 3] = flg;
    }
  }
  constexpr uint8_t coeff_ary[16] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };
  const auto coeff = vld1q_u8(coeff_ary);

#ifdef AARCH32
  const T_UINT TileHeightMax = 20; // configurable
  const T_UINT TileWidthMax = 20; // configurable
  const T_UINT TileHeight = std::min(in_height, TileHeightMax);
  const T_UINT TileWidth = std::min(in_width, TileWidthMax);
  constexpr T_UINT InChUnroll = InTypeBitWidth; // hardcoded, not configurable
  constexpr T_UINT OutChUnroll = 16; // hardcoded, not configurable
  constexpr T_UINT OutChUnroll2 = 32; // hardcoded, not configurable
  constexpr T_UINT InBitChUnroll = 2; // hardcoded, not configurable
  constexpr T_UINT khMax = 5; // hardcoded, not configurable
  constexpr T_UINT kwMax = 5; // hardcoded, not configurable

  const T_UINT row_tile_count = (in_height + TileHeight - 1) / TileHeight;
  const T_UINT col_tile_count = (in_width + TileWidth - 1) / TileWidth;
  const T_UINT out_tile_count = (out_channels + OutChUnroll2 - 1) / OutChUnroll2;
  const T_UINT total_tile_count = row_tile_count * col_tile_count * out_tile_count;
  Measurement::Start("Quantized Conv2D Tiling");
#pragma omp parallel for
  for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
    T_UINT out_ch_high = tile_index % out_tile_count;
    T_UINT col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
    T_UINT row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
    uint32_t out_ts[TileWidthMax*TileWidthMax*OutChUnroll2/OutChUnroll];
    for (unsigned int Om = 0; Om < OutChUnroll2; Om += OutChUnroll) {
      int16_t out_tile[TileHeightMax*TileWidthMax*OutChUnroll];
      for (unsigned int row = 0; row < TileHeight; ++row) {
        for (unsigned int col = 0; col < TileWidth; ++col) {
          for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
            const auto index = row * TileWidth * OutChUnroll
              + col * OutChUnroll
              + out_ch;
            out_tile[index] = 0;
          }
        }
      }
      for (unsigned int in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
        QUANTIZED_PACKED_KERNEL notk[khMax*kwMax*OutChUnroll];
        int16_t notsum[OutChUnroll] = {};
        for (unsigned int out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          notsum[out_ch] = 0;
          for (unsigned int kr = 0; kr < kh; ++kr) {
            for (unsigned int kc = 0; kc < kw; ++kc) {
              const auto notk_index = kr * kw * OutChUnroll
                + kc * OutChUnroll
                + out_ch;
              const auto index = (out_ch_high * OutChUnroll2 + Om + out_ch) * kh * kw * (in_channels / InTypeBitWidth)
                + kr * kw * (in_channels / InTypeBitWidth)
                + kc * (in_channels / InTypeBitWidth)
                + in_ch_high / InTypeBitWidth;
              notk[notk_index] = kernel.data()[index];
              notsum[out_ch] += pop_count(notk[notk_index]);
            }
          }
        }
        for (unsigned int in_bit_ch_high = 0; in_bit_ch_high < in_bitwidth; in_bit_ch_high += InBitChUnroll) {
          tiling_input_elem_t in_tile[(TileHeightMax + khMax - 1)*(TileWidthMax + kwMax - 1)*InBitChUnroll];
          for (unsigned int row = 0; row < TileHeight + kh - 1; ++row) {
            if (row_high + row >= in_height + 2*padding) break;
            for (unsigned int col = 0; col < TileWidth + kw - 1; ++col) {
              if (col_high + col >= in_width + 2*padding) break;
              for (unsigned int in_bit_ch = 0; in_bit_ch < InBitChUnroll; ++in_bit_ch) {
                const auto buf_index = row * (TileWidth + kw - 1) * InBitChUnroll
                  + col * InBitChUnroll
                  + in_bit_ch;
                if (row_high + row < padding || row_high + row >= in_height + padding
                    || col_high + col < padding || col_high + col >= in_width + padding) {
                  in_tile[buf_index] = tiling_input_elem_t(0);
                } else {
                  const auto index = (in_ch_high / InTypeBitWidth) * in_height * in_width * in_bitwidth
                    + (row_high + row - padding) * in_width * in_bitwidth
                    + (col_high + col - padding) * in_bitwidth
                    + (in_bit_ch_high + in_bit_ch);
                  in_tile[buf_index] = input.data()[index];
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
                  const auto notk_index = kr * kw * OutChUnroll
                    + kc * OutChUnroll;
                  uint32x4_t nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index +  0]));
                  uint32x4_t nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index +  4]));
                  uint32x4_t nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index +  8]));
                  uint32x4_t nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index + 12]));
                  uint8x16_t nk08 = vreinterpretq_u8_u32(nk0);
                  uint8x16_t nk18 = vreinterpretq_u8_u32(nk1);
                  uint8x16_t nk28 = vreinterpretq_u8_u32(nk2);
                  uint8x16_t nk38 = vreinterpretq_u8_u32(nk3);
                  const auto in_index = (row + kr) * (TileWidth + kw - 1) * InBitChUnroll
                    + (col + kc) * InBitChUnroll;
                  uint32x4_t in = vdupq_n_u32(in_tile[in_index + 0].Raw());
                  uint8x16_t in8 = vreinterpretq_u8_u32(in);
                  xnorsum00 += vcntq_u8(in8 ^ nk08);
                  xnorsum10 += vcntq_u8(in8 ^ nk18);
                  xnorsum20 += vcntq_u8(in8 ^ nk28);
                  xnorsum30 += vcntq_u8(in8 ^ nk38);
                  in = vdupq_n_u32(in_tile[in_index + 1].Raw());
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
              const auto out_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
              int16x8_t tmp0 = vld1q_s16(&out_tile[out_index + 0]);
              int16x8_t tmp1 = vld1q_s16(&out_tile[out_index + 8]);
              int16x8_t nsum0 = vld1q_s16(&notsum[0]);
              int16x8_t nsum1 = vld1q_s16(&notsum[8]);
              tmp0 += vshlq_s16(sum010 - nsum0, vdupq_n_s16(in_bit_ch_high))
                + vshlq_s16(sum011 - nsum0, vdupq_n_s16(in_bit_ch_high + 1));
              tmp1 += vshlq_s16(sum230 - nsum1, vdupq_n_s16(in_bit_ch_high))
                + vshlq_s16(sum231 - nsum1, vdupq_n_s16(in_bit_ch_high + 1));
              vst1q_s16(&out_tile[out_index + 0], tmp0);
              vst1q_s16(&out_tile[out_index + 8], tmp1);
            }
          }
        }
      }
      if (p.thresholds != nullptr) {
#define APPLY(k) \
  const auto d##k = vld1q_s16(out_tile + buf_index + 8 * k); \
  const auto ts##k = vld4q_s16(buf_th.get() + NUM_OF_A2W1_THRESHOLD * (out_ch_high * OutChUnroll2 + Om + 8 * k)); \
  const auto f##k##0 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[0])) & ts##k.val[3]; \
  const auto f##k##1 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[1])) & ts##k.val[3]; \
  const auto f##k##2 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[2])) & ts##k.val[3]; \
  const auto is_neg##k = vreinterpretq_s16_u16(vcltq_s16(ts##k.val[3], vdupq_n_s16(0))); \
  const auto tmp##k = f##k##0 + f##k##1 + f##k##2 + is_neg##k; \
  const auto m2_##k = vsubq_s16(ts##k.val[3], vdupq_n_s16(2)); \
  const auto is_const##k = vcgeq_s16(m2_##k, vdupq_n_s16(0)); \
  const auto res##k = vreinterpretq_u8_s16(vbslq_s16(is_const##k, m2_##k, tmp##k));

        for (unsigned int row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (unsigned int col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            unsigned int buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            APPLY(0)
            APPLY(1)
            const auto a = vuzpq_u8(res0, res1).val[0]; \
            const auto am = vmulq_u8(vshrq_n_u8(a, 1), coeff); \
            const auto al = vmulq_u8(vandq_u8(a, vdupq_n_u8(0x01)), coeff); \
            const auto bm = vpadd_u8(vget_low_u8(am), vget_high_u8(am)); \
            const auto bl = vpadd_u8(vget_low_u8(al), vget_high_u8(al));
            const auto c = vpadd_u8(bl, bm);
            const auto d = vpadd_u8(c, vdup_n_u8(0));
            unsigned int ts_index = row * TileWidth * 2
                + col * 2 + Om / OutChUnroll;
            out_ts[ts_index] = vget_lane_u32(vreinterpret_u32_u8(d), 0);
          }
        }
#undef APPLY
#undef MAKE_B
      } else {
        for (unsigned int row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (unsigned int col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            unsigned int buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            const auto v0 = vld1q_s16(out_tile + buf_index +  0);
            const auto v1 = vld1q_s16(out_tile + buf_index +  8);
            unsigned int index = out_ch_high * out_height * out_width * OutChUnroll2
                + (row_high + row) * out_width * OutChUnroll2
                + (col_high + col) * OutChUnroll2
                + Om;
            vst1q_s16(p.device_output_buf + index +  0, v0);
            vst1q_s16(p.device_output_buf + index +  8, v1);
          }
        }
      }
    }
    if (p.thresholds != nullptr) {
      const uint8_t table_ary[8] = {
          0, 1, 4, 5, 2, 3, 6, 7
      };
      const auto table = vld1_u8(table_ary);
      for (unsigned int row = 0; row < TileHeight; ++row) {
        if (row_high + row >= out_height) break;
        for (unsigned int col = 0; col < TileWidth; ++col) {
          if (col_high + col >= out_width) break;
          unsigned int buf_index = row * TileWidth * 2
              + col * 2;
          const auto v = vreinterpret_u8_u32(vld1_u32(out_ts + buf_index));
          const auto trnv = vreinterpret_u32_u8(vtbl1_u8(v, table));
          unsigned int index = out_ch_high * out_height * out_width * in_bitwidth
              + (row_high + row) * out_width * in_bitwidth
              + (col_high + col) * in_bitwidth;
          vst1_u32(reinterpret_cast<uint32_t*>(p.device_output_buf) + index, trnv);
        }
      }
    }
  }
#else
  const std::size_t TileHeightMax = 20; // configurable
  const std::size_t TileWidthMax = 20; // configurable
  const std::size_t TileHeight = std::min((std::size_t)in_height, TileHeightMax);
  const std::size_t TileWidth = std::min((std::size_t)in_width + (in_width & 1), TileWidthMax);
  constexpr std::size_t InChUnroll = InTypeBitWidth; // hardcoded, not configurable
  constexpr std::size_t OutChUnroll = 16; // hardcoded, not configurable
  constexpr std::size_t OutChUnroll2 = 32; // hardcoded, not configurable
  constexpr std::size_t InBitChUnroll = 2; // hardcoded, not configurable
  constexpr std::size_t ColUnroll = 2; // hardcoded, not configurable
  constexpr std::size_t khMax = 5; // hardcoded, not configurable
  constexpr std::size_t kwMax = 5; // hardcoded, not configurable

  const std::size_t kh_s = cp.kernel_height;
  const std::size_t kw_s = cp.kernel_width;
  const std::size_t row_tile_count = (in_height + TileHeight - 1) / TileHeight;
  const std::size_t col_tile_count = (in_width + TileWidth - 1) / TileWidth;
  const std::size_t out_tile_count = (out_channels + OutChUnroll2 - 1) / OutChUnroll2;
  const std::size_t total_tile_count = row_tile_count * col_tile_count * out_tile_count;
  Measurement::Start("Quantized Conv2D Tiling");
#pragma omp parallel for
  for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
    std::size_t out_ch_high = tile_index % out_tile_count;
    std::size_t col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
    std::size_t row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
    uint32_t out_ts[TileHeightMax*TileWidthMax*OutChUnroll2/OutChUnroll];
    for (std::size_t Om = 0; Om < OutChUnroll2; Om += OutChUnroll) {
      int16_t out_tile[TileHeightMax*TileWidthMax*OutChUnroll];
      for (std::size_t row = 0; row < TileHeight; ++row) {
        for (std::size_t col = 0; col < TileWidth; ++col) {
          for (std::size_t out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
            std::size_t index = row * TileWidth * OutChUnroll
                + col * OutChUnroll
                + out_ch;
            out_tile[index] = 0;
          }
        }
      }
      for (std::size_t in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
        QUANTIZED_PACKED_KERNEL notk[khMax*kwMax*OutChUnroll];
        int16_t notsum[OutChUnroll] = {};
        for (std::size_t out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
          notsum[out_ch] = 0;
          for (std::size_t kr = 0; kr < kh; ++kr) {
            for (std::size_t kc = 0; kc < kw; ++kc) {
              const std::size_t index = (out_ch_high * OutChUnroll2 + Om + out_ch) * kh * kw * (in_channels / InTypeBitWidth)
                + kr * kw * (in_channels / InTypeBitWidth)
                + kc * (in_channels / InTypeBitWidth)
                + in_ch_high / InTypeBitWidth;
              const std::size_t notk_index = kr * kw * OutChUnroll
                  + kc * OutChUnroll
                  + out_ch;
              notk[notk_index] = kernel.data()[index];
              notsum[out_ch] += pop_count(notk[notk_index]);
            }
          }
        }
        for (std::size_t in_bit_ch_high = 0; in_bit_ch_high < in_bitwidth; in_bit_ch_high += InBitChUnroll) {
          tiling_input_elem_t in_tile[(TileHeightMax + khMax - 1)*(TileWidthMax + kwMax - 1)*InBitChUnroll];
          for (std::size_t row = 0; row < TileHeight + kh_s - 1; ++row) {
            if (row_high + row >= in_height + 2*padding) break;
            for (std::size_t col = 0; col < TileWidth + kw_s - 1; ++col) {
              if (col_high + col >= in_width + 2*padding) break;
              const auto in_tile_index = row * (TileWidth + kw_s - 1) * InBitChUnroll
                  + col * InBitChUnroll;
              if (row_high + row < padding || row_high + row >= in_height + padding
                  || col_high + col < padding || col_high + col >= in_width + padding) {
                vst1_u32(reinterpret_cast<uint32_t*>(in_tile + in_tile_index), vdup_n_u32(0));
              } else {
                const auto index = (in_ch_high / InTypeBitWidth) * in_height * in_width * in_bitwidth
                  + (row_high + row - padding) * in_width * in_bitwidth
                  + (col_high + col - padding) * in_bitwidth
                  + in_bit_ch_high;
                const auto v = vld1_u32(reinterpret_cast<uint32_t*>(input.data() + index));
                vst1_u32(reinterpret_cast<uint32_t*>(in_tile + in_tile_index), v);
              }
            }
          }
          for (std::size_t row = 0; row < TileHeight; ++row) {
            for (std::size_t col = 0; col < TileWidth; col += ColUnroll) {
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
              for (std::size_t kr = 0; kr < kh_s; ++kr) {
                const std::size_t in_index = (row + kr) * (TileWidth + kw_s - 1) * InBitChUnroll
                    + col * InBitChUnroll;
                uint32x4_t inl0 = vdupq_n_u32(in_tile[in_index + 0].Raw());
                uint32x4_t inh0 = vdupq_n_u32(in_tile[in_index + 1].Raw());
                uint8x16_t inl08 = vreinterpretq_u8_u32(inl0);
                uint8x16_t inh08 = vreinterpretq_u8_u32(inh0);
                for (std::size_t kc = 0; kc < kw_s; ++kc) {
                  const std::size_t nk_index = kr * kw_s * OutChUnroll
                      + kc * OutChUnroll;
                  uint32x4_t nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index +  0]));
                  uint32x4_t nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index +  4]));
                  uint32x4_t nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index +  8]));
                  uint32x4_t nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index + 12]));
                  uint8x16_t nk08 = vreinterpretq_u8_u32(nk0);
                  uint8x16_t nk18 = vreinterpretq_u8_u32(nk1);
                  uint8x16_t nk28 = vreinterpretq_u8_u32(nk2);
                  uint8x16_t nk38 = vreinterpretq_u8_u32(nk3);
                  uint32x4_t inl1 = vdupq_n_u32(in_tile[in_index + (kc + 1) * InBitChUnroll + 0].Raw());
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
                  uint32x4_t inh1 = vdupq_n_u32(in_tile[in_index + (kc + 1) * InBitChUnroll + 1].Raw());
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
              std::size_t out_index = row * TileWidth * OutChUnroll
                  + col * OutChUnroll;
              int16x8_t tmp00 = vld1q_s16(&out_tile[out_index + 0 * OutChUnroll + 0]);
              int16x8_t tmp01 = vld1q_s16(&out_tile[out_index + 0 * OutChUnroll + 8]);
              int16x8_t tmp10 = vld1q_s16(&out_tile[out_index + 1 * OutChUnroll + 0]);
              int16x8_t tmp11 = vld1q_s16(&out_tile[out_index + 1 * OutChUnroll + 8]);
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
              vst1q_s16(&out_tile[out_index + 0 * OutChUnroll + 0], tmp00);
              vst1q_s16(&out_tile[out_index + 0 * OutChUnroll + 8], tmp01);
              vst1q_s16(&out_tile[out_index + 1 * OutChUnroll + 0], tmp10);
              vst1q_s16(&out_tile[out_index + 1 * OutChUnroll + 8], tmp11);
            }
          }
        }
      }
      if (p.thresholds != nullptr) {
#define APPLY(k) \
  const auto d##k = vld1q_s16(out_tile + buf_index + 8 * k); \
  const auto ts##k = vld4q_s16(buf_th.get() + NUM_OF_A2W1_THRESHOLD * (out_ch_high * OutChUnroll2 + Om + 8 * k)); \
  const auto f##k##0 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[0])) & ts##k.val[3]; \
  const auto f##k##1 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[1])) & ts##k.val[3]; \
  const auto f##k##2 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[2])) & ts##k.val[3]; \
  const auto is_neg##k = vreinterpretq_s16_u16(vcltq_s16(ts##k.val[3], vdupq_n_s16(0))); \
  const auto tmp##k = f##k##0 + f##k##1 + f##k##2 + is_neg##k; \
  const auto m2_##k = vsubq_s16(ts##k.val[3], vdupq_n_s16(2)); \
  const auto is_const##k = vcgeq_s16(m2_##k, vdupq_n_s16(0)); \
  const auto res##k = vreinterpretq_u8_s16(vbslq_s16(is_const##k, m2_##k, tmp##k));

        for (std::size_t row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (std::size_t col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            std::size_t buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            APPLY(0)
            APPLY(1)
            const auto a = vuzpq_u8(res0, res1).val[0]; \
            const auto am = vmulq_u8(vshrq_n_u8(a, 1), coeff); \
            const auto al = vmulq_u8(vandq_u8(a, vdupq_n_u8(0x01)), coeff); \
            const auto bm = vpadd_u8(vget_low_u8(am), vget_high_u8(am)); \
            const auto bl = vpadd_u8(vget_low_u8(al), vget_high_u8(al));
            const auto c = vpadd_u8(bl, bm);
            const auto d = vpadd_u8(c, vdup_n_u8(0));
            std::size_t ts_index = row * TileWidth * 2
                + col * 2 + Om / OutChUnroll;
            out_ts[ts_index] = vget_lane_u32(vreinterpret_u32_u8(d), 0);
          }
        }
#undef APPLY
#undef MAKE_B
      } else {
        for (std::size_t row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (std::size_t col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            std::size_t buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            const auto v0 = vld1q_s16(out_tile + buf_index +  0);
            const auto v1 = vld1q_s16(out_tile + buf_index +  8);
            std::size_t index = out_ch_high * out_height * out_width * OutChUnroll2
                + (row_high + row) * out_width * OutChUnroll2
                + (col_high + col) * OutChUnroll2
                + Om;
            vst1q_s16(p.device_output_buf + index +  0, v0);
            vst1q_s16(p.device_output_buf + index +  8, v1);
          }
        }
      }
    }
    if (p.thresholds != nullptr) {
      const uint8_t table_ary[8] = {
          0, 1, 4, 5, 2, 3, 6, 7
      };
      const auto table = vld1_u8(table_ary);
      for (std::size_t row = 0; row < TileHeight; ++row) {
        if (row_high + row >= out_height) break;
        for (std::size_t col = 0; col < TileWidth; ++col) {
          if (col_high + col >= out_width) break;
          std::size_t buf_index = row * TileWidth * 2
              + col * 2;
          const auto v = vreinterpret_u8_u32(vld1_u32(out_ts + buf_index));
          const auto trnv = vreinterpret_u32_u8(vtbl1_u8(v, table));
          std::size_t index = out_ch_high * out_height * out_width * in_bitwidth
              + (row_high + row) * out_width * in_bitwidth
              + (col_high + col) * in_bitwidth;
          vst1_u32(reinterpret_cast<uint32_t*>(p.device_output_buf) + index, trnv);
        }
      }
    }
  }
#endif
  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

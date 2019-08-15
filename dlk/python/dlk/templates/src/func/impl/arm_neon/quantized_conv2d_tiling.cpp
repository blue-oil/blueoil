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
#include "func/impl/quantized_conv2d_tiling.h"
#include "time_measurement.h"

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

  Measurement::Start("Quantized Conv2D Tiling");
  if (p.thresholds != nullptr) {
    for (T_UINT i = 0; i < out_channels; i += 8) {
      const auto v = vld4q_s16(p.thresholds + NUM_OF_A2W1_THRESHOLD * i);
      const auto is_neg = vreinterpretq_s16_u16(vmvnq_u16(vcgeq_s16(v.val[3], vdupq_n_s16(0))));
      int16x8x4_t res;
      res.val[0] = vsubq_s16(v.val[0], is_neg);
      res.val[1] = vsubq_s16(v.val[1], is_neg);
      res.val[2] = vsubq_s16(v.val[2], is_neg);
      res.val[3] = v.val[3];
      vst4q_s16(buf_th.get() + NUM_OF_A2W1_THRESHOLD * i, res);
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
#pragma omp parallel for
  for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
    T_UINT out_ch_high = tile_index % out_tile_count;
    T_UINT col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
    T_UINT row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
    uint32_t out_ts[TileWidthMax*TileWidthMax*OutChUnroll2/OutChUnroll];
    for (unsigned int Om = 0; Om < OutChUnroll2; Om += OutChUnroll) {
      BIN_CONV_OUTPUT out_tile[TileHeightMax*TileWidthMax*OutChUnroll];
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
        BIN_CONV_OUTPUT notsum[OutChUnroll] = {};
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
              const auto in_tile_index = row * (TileWidth + kw - 1) * InBitChUnroll
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
          for (unsigned int row = 0; row < TileHeight; ++row) {
            for (unsigned int col = 0; col < TileWidth; ++col) {
              auto xnorsum00 = vdupq_n_u8(0);
              auto xnorsum01 = vdupq_n_u8(0);
              auto xnorsum10 = vdupq_n_u8(0);
              auto xnorsum11 = vdupq_n_u8(0);
              auto xnorsum20 = vdupq_n_u8(0);
              auto xnorsum21 = vdupq_n_u8(0);
              auto xnorsum30 = vdupq_n_u8(0);
              auto xnorsum31 = vdupq_n_u8(0);
              for (unsigned int kr = 0; kr < kh; ++kr) {
                for (unsigned int kc = 0; kc < kw; ++kc) {
                  const auto notk_index = kr * kw * OutChUnroll
                    + kc * OutChUnroll;
                  const auto nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index +  0]));
                  const auto nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index +  4]));
                  const auto nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index +  8]));
                  const auto nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[notk_index + 12]));
                  const auto nk08 = vreinterpretq_u8_u32(nk0);
                  const auto nk18 = vreinterpretq_u8_u32(nk1);
                  const auto nk28 = vreinterpretq_u8_u32(nk2);
                  const auto nk38 = vreinterpretq_u8_u32(nk3);
                  const auto in_index = (row + kr) * (TileWidth + kw - 1) * InBitChUnroll
                    + (col + kc) * InBitChUnroll;
                  const auto in0 = vdupq_n_u32(in_tile[in_index + 0].Raw());
                  const auto in08 = vreinterpretq_u8_u32(in0);
                  xnorsum00 += vcntq_u8(in08 ^ nk08);
                  xnorsum10 += vcntq_u8(in08 ^ nk18);
                  xnorsum20 += vcntq_u8(in08 ^ nk28);
                  xnorsum30 += vcntq_u8(in08 ^ nk38);
                  const auto in1 = vdupq_n_u32(in_tile[in_index + 1].Raw());
                  const auto in18 = vreinterpretq_u8_u32(in1);
                  xnorsum01 += vcntq_u8(in18 ^ nk08);
                  xnorsum11 += vcntq_u8(in18 ^ nk18);
                  xnorsum21 += vcntq_u8(in18 ^ nk28);
                  xnorsum31 += vcntq_u8(in18 ^ nk38);
                }
              }
              const auto psum000 = vpaddlq_u8(xnorsum00);
              const auto psum010 = vpaddlq_u8(xnorsum10);
              const auto psum020 = vpaddlq_u8(xnorsum20);
              const auto psum030 = vpaddlq_u8(xnorsum30);
              const auto psum001 = vpaddlq_u8(xnorsum01);
              const auto psum011 = vpaddlq_u8(xnorsum11);
              const auto psum021 = vpaddlq_u8(xnorsum21);
              const auto psum031 = vpaddlq_u8(xnorsum31);
              const auto psum100 = vpadd_u16(vget_low_u16(psum000), vget_high_u16(psum000));
              const auto psum110 = vpadd_u16(vget_low_u16(psum010), vget_high_u16(psum010));
              const auto psum120 = vpadd_u16(vget_low_u16(psum020), vget_high_u16(psum020));
              const auto psum130 = vpadd_u16(vget_low_u16(psum030), vget_high_u16(psum030));
              const auto psum101 = vpadd_u16(vget_low_u16(psum001), vget_high_u16(psum001));
              const auto psum111 = vpadd_u16(vget_low_u16(psum011), vget_high_u16(psum011));
              const auto psum121 = vpadd_u16(vget_low_u16(psum021), vget_high_u16(psum021));
              const auto psum131 = vpadd_u16(vget_low_u16(psum031), vget_high_u16(psum031));
              const auto usum010 = vcombine_u16(psum100, psum110);
              const auto usum230 = vcombine_u16(psum120, psum130);
              const auto usum011 = vcombine_u16(psum101, psum111);
              const auto usum231 = vcombine_u16(psum121, psum131);
              const auto sum010 = vreinterpretq_s16_u16(usum010);
              const auto sum230 = vreinterpretq_s16_u16(usum230);
              const auto sum011 = vreinterpretq_s16_u16(usum011);
              const auto sum231 = vreinterpretq_s16_u16(usum231);
              const auto out_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
              auto tmp0 = vld1q_s16(&out_tile[out_index + 0]);
              auto tmp1 = vld1q_s16(&out_tile[out_index + 8]);
              const auto nsum0 = vld1q_s16(&notsum[0]);
              const auto nsum1 = vld1q_s16(&notsum[8]);
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
            const auto buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            APPLY(0)
            APPLY(1)
            const auto a = vuzpq_u8(res0, res1).val[0];
            const auto am = vmulq_u8(vshrq_n_u8(a, 1), coeff);
            const auto al = vmulq_u8(vandq_u8(a, vdupq_n_u8(0x01)), coeff);
            const auto bm = vpadd_u8(vget_low_u8(am), vget_high_u8(am));
            const auto bl = vpadd_u8(vget_low_u8(al), vget_high_u8(al));
            const auto c = vpadd_u8(bl, bm);
            const auto d = vpadd_u8(c, vdup_n_u8(0));
            const auto ts_index = row * TileWidth * 2
                + col * 2 + Om / OutChUnroll;
            out_ts[ts_index] = vget_lane_u32(vreinterpret_u32_u8(d), 0);
          }
        }
#undef APPLY
      } else {
        for (unsigned int row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (unsigned int col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            const auto buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            const auto v0 = vld1q_s16(out_tile + buf_index +  0);
            const auto v1 = vld1q_s16(out_tile + buf_index +  8);
            const auto index = out_ch_high * out_height * out_width * OutChUnroll2
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
          const auto buf_index = row * TileWidth * 2
              + col * 2;
          const auto v = vreinterpret_u8_u32(vld1_u32(out_ts + buf_index));
          const auto trnv = vreinterpret_u32_u8(vtbl1_u8(v, table));
          const auto index = out_ch_high * out_height * out_width * in_bitwidth
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
#pragma omp parallel for
  for (std::size_t tile_index = 0; tile_index < total_tile_count; ++tile_index) {
    std::size_t out_ch_high = tile_index % out_tile_count;
    std::size_t col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
    std::size_t row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
    uint32_t out_ts[TileHeightMax*TileWidthMax*OutChUnroll2/OutChUnroll];
    for (std::size_t Om = 0; Om < OutChUnroll2; Om += OutChUnroll) {
      BIN_CONV_OUTPUT out_tile[TileHeightMax*TileWidthMax*OutChUnroll];
      for (std::size_t row = 0; row < TileHeight; ++row) {
        for (std::size_t col = 0; col < TileWidth; ++col) {
          for (std::size_t out_ch = 0; out_ch < OutChUnroll; ++out_ch) {
            const auto index = row * TileWidth * OutChUnroll
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
          for (std::size_t kr = 0; kr < kh_s; ++kr) {
            for (std::size_t kc = 0; kc < kw_s; ++kc) {
              const auto index = (out_ch_high * OutChUnroll2 + Om + out_ch) * kh_s * kw_s * (in_channels / InTypeBitWidth)
                + kr * kw_s * (in_channels / InTypeBitWidth)
                + kc * (in_channels / InTypeBitWidth)
                + in_ch_high / InTypeBitWidth;
              const auto notk_index = kr * kw_s * OutChUnroll
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
              auto xnorsum000 = vdupq_n_u8(0);
              auto xnorsum001 = vdupq_n_u8(0);
              auto xnorsum010 = vdupq_n_u8(0);
              auto xnorsum011 = vdupq_n_u8(0);
              auto xnorsum020 = vdupq_n_u8(0);
              auto xnorsum021 = vdupq_n_u8(0);
              auto xnorsum030 = vdupq_n_u8(0);
              auto xnorsum031 = vdupq_n_u8(0);
              auto xnorsum100 = vdupq_n_u8(0);
              auto xnorsum101 = vdupq_n_u8(0);
              auto xnorsum110 = vdupq_n_u8(0);
              auto xnorsum111 = vdupq_n_u8(0);
              auto xnorsum120 = vdupq_n_u8(0);
              auto xnorsum121 = vdupq_n_u8(0);
              auto xnorsum130 = vdupq_n_u8(0);
              auto xnorsum131 = vdupq_n_u8(0);
              for (std::size_t kr = 0; kr < kh_s; ++kr) {
                const auto in_index = (row + kr) * (TileWidth + kw_s - 1) * InBitChUnroll
                    + col * InBitChUnroll;
                const auto inl0 = vdupq_n_u32(in_tile[in_index + 0].Raw());
                const auto inh0 = vdupq_n_u32(in_tile[in_index + 1].Raw());
                auto inl08 = vreinterpretq_u8_u32(inl0);
                auto inh08 = vreinterpretq_u8_u32(inh0);
                for (std::size_t kc = 0; kc < kw_s; ++kc) {
                  const auto nk_index = kr * kw_s * OutChUnroll
                      + kc * OutChUnroll;
                  const auto nk0 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index +  0]));
                  const auto nk1 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index +  4]));
                  const auto nk2 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index +  8]));
                  const auto nk3 = vld1q_u32(reinterpret_cast<uint32_t*>(&notk[nk_index + 12]));
                  const auto nk08 = vreinterpretq_u8_u32(nk0);
                  const auto nk18 = vreinterpretq_u8_u32(nk1);
                  const auto nk28 = vreinterpretq_u8_u32(nk2);
                  const auto nk38 = vreinterpretq_u8_u32(nk3);
                  const auto inl1 = vdupq_n_u32(in_tile[in_index + (kc + 1) * InBitChUnroll + 0].Raw());
                  const auto inl18 = vreinterpretq_u8_u32(inl1);
                  xnorsum000 += vcntq_u8(inl08 ^ nk08);
                  xnorsum010 += vcntq_u8(inl08 ^ nk18);
                  xnorsum020 += vcntq_u8(inl08 ^ nk28);
                  xnorsum030 += vcntq_u8(inl08 ^ nk38);
                  xnorsum100 += vcntq_u8(inl18 ^ nk08);
                  xnorsum110 += vcntq_u8(inl18 ^ nk18);
                  xnorsum120 += vcntq_u8(inl18 ^ nk28);
                  xnorsum130 += vcntq_u8(inl18 ^ nk38);
                  inl08 = inl18;
                  const auto inh1 = vdupq_n_u32(in_tile[in_index + (kc + 1) * InBitChUnroll + 1].Raw());
                  const auto inh18 = vreinterpretq_u8_u32(inh1);
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
              const auto psum0000 = vpaddlq_u8(xnorsum000);
              const auto psum0010 = vpaddlq_u8(xnorsum010);
              const auto psum0020 = vpaddlq_u8(xnorsum020);
              const auto psum0030 = vpaddlq_u8(xnorsum030);
              const auto psum0001 = vpaddlq_u8(xnorsum001);
              const auto psum0011 = vpaddlq_u8(xnorsum011);
              const auto psum0021 = vpaddlq_u8(xnorsum021);
              const auto psum0031 = vpaddlq_u8(xnorsum031);
              const auto psum0100 = vpaddlq_u8(xnorsum100);
              const auto psum0110 = vpaddlq_u8(xnorsum110);
              const auto psum0120 = vpaddlq_u8(xnorsum120);
              const auto psum0130 = vpaddlq_u8(xnorsum130);
              const auto psum0101 = vpaddlq_u8(xnorsum101);
              const auto psum0111 = vpaddlq_u8(xnorsum111);
              const auto psum0121 = vpaddlq_u8(xnorsum121);
              const auto psum0131 = vpaddlq_u8(xnorsum131);
              const auto psum1000 = vreinterpretq_u16_u32(vpaddlq_u16(psum0000));
              const auto psum1010 = vreinterpretq_u16_u32(vpaddlq_u16(psum0010));
              const auto psum1020 = vreinterpretq_u16_u32(vpaddlq_u16(psum0020));
              const auto psum1030 = vreinterpretq_u16_u32(vpaddlq_u16(psum0030));
              const auto psum1001 = vreinterpretq_u16_u32(vpaddlq_u16(psum0001));
              const auto psum1011 = vreinterpretq_u16_u32(vpaddlq_u16(psum0011));
              const auto psum1021 = vreinterpretq_u16_u32(vpaddlq_u16(psum0021));
              const auto psum1031 = vreinterpretq_u16_u32(vpaddlq_u16(psum0031));
              const auto psum1100 = vreinterpretq_u16_u32(vpaddlq_u16(psum0100));
              const auto psum1110 = vreinterpretq_u16_u32(vpaddlq_u16(psum0110));
              const auto psum1120 = vreinterpretq_u16_u32(vpaddlq_u16(psum0120));
              const auto psum1130 = vreinterpretq_u16_u32(vpaddlq_u16(psum0130));
              const auto psum1101 = vreinterpretq_u16_u32(vpaddlq_u16(psum0101));
              const auto psum1111 = vreinterpretq_u16_u32(vpaddlq_u16(psum0111));
              const auto psum1121 = vreinterpretq_u16_u32(vpaddlq_u16(psum0121));
              const auto psum1131 = vreinterpretq_u16_u32(vpaddlq_u16(psum0131));
              const auto usum0010 = vuzpq_u16(psum1000, psum1010).val[0];
              const auto usum0230 = vuzpq_u16(psum1020, psum1030).val[0];
              const auto usum0011 = vuzpq_u16(psum1001, psum1011).val[0];
              const auto usum0231 = vuzpq_u16(psum1021, psum1031).val[0];
              const auto usum1010 = vuzpq_u16(psum1100, psum1110).val[0];
              const auto usum1230 = vuzpq_u16(psum1120, psum1130).val[0];
              const auto usum1011 = vuzpq_u16(psum1101, psum1111).val[0];
              const auto usum1231 = vuzpq_u16(psum1121, psum1131).val[0];
              const auto sum0010 = vreinterpretq_s16_u16(usum0010);
              const auto sum0230 = vreinterpretq_s16_u16(usum0230);
              const auto sum0011 = vreinterpretq_s16_u16(usum0011);
              const auto sum0231 = vreinterpretq_s16_u16(usum0231);
              const auto sum1010 = vreinterpretq_s16_u16(usum1010);
              const auto sum1230 = vreinterpretq_s16_u16(usum1230);
              const auto sum1011 = vreinterpretq_s16_u16(usum1011);
              const auto sum1231 = vreinterpretq_s16_u16(usum1231);
              const auto out_index = row * TileWidth * OutChUnroll
                  + col * OutChUnroll;
              auto tmp00 = vld1q_s16(&out_tile[out_index + 0 * OutChUnroll + 0]);
              auto tmp01 = vld1q_s16(&out_tile[out_index + 0 * OutChUnroll + 8]);
              auto tmp10 = vld1q_s16(&out_tile[out_index + 1 * OutChUnroll + 0]);
              auto tmp11 = vld1q_s16(&out_tile[out_index + 1 * OutChUnroll + 8]);
              const auto nsum0 = vld1q_s16(&notsum[ 0]);
              const auto nsum1 = vld1q_s16(&notsum[ 8]);
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
            const auto buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            APPLY(0)
            APPLY(1)
            const auto a = vuzpq_u8(res0, res1).val[0];
            const auto am = vmulq_u8(vshrq_n_u8(a, 1), coeff);
            const auto al = vmulq_u8(vandq_u8(a, vdupq_n_u8(0x01)), coeff);
            const auto bm = vpadd_u8(vget_low_u8(am), vget_high_u8(am));
            const auto bl = vpadd_u8(vget_low_u8(al), vget_high_u8(al));
            const auto c = vpadd_u8(bl, bm);
            const auto d = vpadd_u8(c, vdup_n_u8(0));
            const auto ts_index = row * TileWidth * 2
                + col * 2 + Om / OutChUnroll;
            out_ts[ts_index] = vget_lane_u32(vreinterpret_u32_u8(d), 0);
          }
        }
#undef APPLY
      } else {
        for (std::size_t row = 0; row < TileHeight; ++row) {
          if (row_high + row >= out_height) break;
          for (std::size_t col = 0; col < TileWidth; ++col) {
            if (col_high + col >= out_width) break;
            const auto buf_index = row * TileWidth * OutChUnroll
                + col * OutChUnroll;
            const auto v0 = vld1q_s16(out_tile + buf_index +  0);
            const auto v1 = vld1q_s16(out_tile + buf_index +  8);
            const auto index = out_ch_high * out_height * out_width * OutChUnroll2
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
          const auto buf_index = row * TileWidth * 2
              + col * 2;
          const auto v = vreinterpret_u8_u32(vld1_u32(out_ts + buf_index));
          const auto trnv = vreinterpret_u32_u8(vtbl1_u8(v, table));
          const auto index = out_ch_high * out_height * out_width * in_bitwidth
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

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

#include <x86intrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dlk {

namespace impl {

static const auto buf_th = std::make_unique<QUANTIZED_PACKED[]>(MAX_SIZE_QOUTPUTS_PER_LAYER);
static const auto buf_non_th = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_SIZE_OUTPUTS_PER_LAYER);
alignas(32) static int16_t nksum_ary[MAX_SIZE_QKERNELS_PER_LAYER];
alignas(32) static uint16_t nk[MAX_SIZE_QKERNELS_PER_LAYER*2];

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

  //assert(kh * kw < 32);
  assert(in_height * in_width == out_height * out_width);
  assert((in_channels % InTypeBitWidth) == 0);

  Measurement::Start("Quantized Conv2D Tiling");
  const auto mask4 = _mm256_set1_epi8(0x0F);
  const auto popc_table = _mm256_setr_epi8(
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
  );
  const auto vone = _mm256_set1_epi8(0x01);
  if (kh == 1 && kw == 1) {
    constexpr T_UINT InChUnroll = InTypeBitWidth; // hardcoded, not configurable
    constexpr T_UINT OutChUnroll = 16; // hardcoded, not configurable
    constexpr T_UINT InBitChUnroll = 2; // hardcoded, not configurable
    constexpr T_UINT ColUnroll = 4; // hardcoded, not configurable
    const auto row_tile_count = in_height;
    const auto col_tile_count = (in_width + ColUnroll - 1) / ColUnroll;
    const auto out_tile_count = (out_channels + OutChUnroll - 1) / OutChUnroll;
    const auto total_tile_count = row_tile_count * col_tile_count * out_tile_count;
    for (T_UINT i = 0; i < out_channels; ++i) {
      nksum_ary[i] = 0;
    }
    for (T_UINT i = 0; i < out_channels; i += OutChUnroll) {
      for (T_UINT j = 0; j < in_channels / InTypeBitWidth; ++j) {
        for (T_UINT k = 0; k < OutChUnroll; ++k) {
          const auto nk_tmp
            = kernel.data()[(i+k) * (in_channels / InTypeBitWidth) + j].Raw();
          nk[i * (in_channels / InTypeBitWidth * 2) + (j * 2) * OutChUnroll + k]
            = nk_tmp;
          nk[i * (in_channels / InTypeBitWidth * 2) + (j * 2 + 1) * OutChUnroll + k]
            = nk_tmp >> 16;
          nksum_ary[i + k]
            += __builtin_popcount(nk_tmp) * 3;
        }
      }
    }
    const auto voffset = _mm256_sub_epi64(_mm256_setr_epi64x(0, 1, 2, 3), _mm256_set1_epi64x(in_width));
#pragma omp parallel for schedule(guided)
    for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
      const auto out_ch_high = tile_index % out_tile_count * OutChUnroll;
      const auto col = (tile_index / out_tile_count) % col_tile_count * ColUnroll;
      const auto row = tile_index / (out_tile_count * col_tile_count);
      auto xnorsum00 = _mm256_setzero_si256();
      auto xnorsum01 = _mm256_setzero_si256();
      auto xnorsum10 = _mm256_setzero_si256();
      auto xnorsum11 = _mm256_setzero_si256();
      auto xnorsum20 = _mm256_setzero_si256();
      auto xnorsum21 = _mm256_setzero_si256();
      auto xnorsum30 = _mm256_setzero_si256();
      auto xnorsum31 = _mm256_setzero_si256();
      for (unsigned int in_ch_high = 0; in_ch_high < in_channels; in_ch_high += InTypeBitWidth) {
        const auto nk_index = out_ch_high * (in_channels / InTypeBitWidth * 2)
          + (in_ch_high / InTypeBitWidth * 2) * OutChUnroll;
        const auto nk0 = _mm256_load_si256(reinterpret_cast<__m256i*>(nk + nk_index + 0 * OutChUnroll));
        const auto nk1 = _mm256_load_si256(reinterpret_cast<__m256i*>(nk + nk_index + 1 * OutChUnroll));
        const auto vcol = _mm256_add_epi64(_mm256_set1_epi64x(col), voffset);
        const auto loadmask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), vcol);
        const auto in_index = (in_ch_high / InTypeBitWidth) * in_height * in_width * in_bitwidth
          + row * in_width * in_bitwidth
          + col * in_bitwidth;
        const auto in = _mm256_maskload_epi64(reinterpret_cast<const long long*>(input.data() + in_index), loadmask);
        const auto in_lo = _mm256_castsi256_si128(in);
        const auto in_hi = _mm256_extracti128_si256(in, 1);
        const auto in000 = _mm256_broadcastw_epi16(in_lo);
        const auto in001 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo,  4));
        const auto in010 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo,  8));
        const auto in011 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo, 12));
        const auto in020 = _mm256_broadcastw_epi16(in_hi);
        const auto in021 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi,  4));
        const auto in030 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi,  8));
        const auto in031 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi, 12));
#define BINDP(i, j, k) \
  do { \
    const auto xnor = in##i##j##k ^ nk##i; \
    const auto l4 = mask4 & xnor; \
    const auto popc_l4 = _mm256_shuffle_epi8(popc_table, l4); \
    const auto h4 = mask4 & _mm256_srli_epi32(xnor, 4); \
    const auto popc_h4 = _mm256_shuffle_epi8(popc_table, h4); \
    const auto cnt = _mm256_add_epi8(popc_l4, popc_h4); \
    const auto cnt16 = _mm256_maddubs_epi16(cnt, vone); \
    xnorsum##j##k = _mm256_add_epi16(xnorsum##j##k, cnt16); \
  } while(0)
        BINDP(0, 0, 0);
        BINDP(0, 0, 1);
        BINDP(0, 1, 0);
        BINDP(0, 1, 1);
        BINDP(0, 2, 0);
        BINDP(0, 2, 1);
        BINDP(0, 3, 0);
        BINDP(0, 3, 1);
        const auto in100 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo,  2));
        const auto in101 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo,  6));
        const auto in110 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo, 10));
        const auto in111 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_lo, 14));
        const auto in120 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi,  2));
        const auto in121 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi,  6));
        const auto in130 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi, 10));
        const auto in131 = _mm256_broadcastw_epi16(_mm_bsrli_si128(in_hi, 14));
        BINDP(1, 0, 0);
        BINDP(1, 0, 1);
        BINDP(1, 1, 0);
        BINDP(1, 1, 1);
        BINDP(1, 2, 0);
        BINDP(1, 2, 1);
        BINDP(1, 3, 0);
        BINDP(1, 3, 1);
#undef BINDP
      }
      const auto nksum = _mm256_load_si256(reinterpret_cast<__m256i*>(nksum_ary + out_ch_high));
      const auto out_index = row * out_width * out_channels
          + col * out_channels
          + out_ch_high;
#define CALC(j) \
  if (col + j >= out_width) continue; \
  do { \
    const auto shifted = _mm256_slli_epi16(xnorsum##j##1, 1); \
    const auto tmp = _mm256_add_epi16(xnorsum##j##0, shifted); \
    const auto res = _mm256_sub_epi16(tmp, nksum); \
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p.device_output_buf + out_index + j * out_channels), res); \
  } while (0)
      CALC(0);
      CALC(1);
      CALC(2);
      CALC(3);
    }
  } else {
    constexpr T_UINT InChUnroll = InTypeBitWidth; // hardcoded, not configurable
    constexpr T_UINT OutChUnroll = 16; // hardcoded, not configurable
    constexpr T_UINT InBitChUnroll = 2; // hardcoded, not configurable
    constexpr T_UINT ColUnroll = 2; // hardcoded, not configurable
    const T_UINT TileHeight = std::min(in_height, T_UINT(16)); // configurable
    const T_UINT TileWidth = std::min(in_width + (in_width & 1), T_UINT(16)); // configurable
    const T_UINT row_tile_count = (in_height + TileHeight - 1) / TileHeight;
    const T_UINT col_tile_count = (in_width + TileWidth - 1) / TileWidth;
    const T_UINT out_tile_count = (out_channels + OutChUnroll - 1) / OutChUnroll;
    const T_UINT total_tile_count = row_tile_count * col_tile_count * out_tile_count;
#pragma omp parallel for schedule(guided)
    for (T_UINT tile_index = 0; tile_index < total_tile_count; ++tile_index) {
      const auto out_ch_high = tile_index % out_tile_count * OutChUnroll;
      const auto col_high = (tile_index / out_tile_count) % col_tile_count * TileWidth;
      const auto row_high = tile_index / (out_tile_count * col_tile_count) * TileHeight;
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
                + (in_ch_high / InTypeBitWidth);
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
              auto xnorsum000 = _mm256_setzero_si256();
              auto xnorsum001 = _mm256_setzero_si256();
              auto xnorsum010 = _mm256_setzero_si256();
              auto xnorsum011 = _mm256_setzero_si256();
              auto xnorsum100 = _mm256_setzero_si256();
              auto xnorsum101 = _mm256_setzero_si256();
              auto xnorsum110 = _mm256_setzero_si256();
              auto xnorsum111 = _mm256_setzero_si256();
              for (unsigned int kr = 0; kr < kh; ++kr) {
                auto in00 = _mm256_set1_epi32(in_tile[row + kr][col][0].Raw());
                auto in10 = _mm256_set1_epi32(in_tile[row + kr][col][1].Raw());
                for (unsigned int kc = 0; kc < kw; ++kc) {
                  const auto nk0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&notk[kr][kc][ 0]));
                  const auto nk1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&notk[kr][kc][ 8]));
                  const auto in01 = _mm256_set1_epi32(in_tile[row + kr][col + kc + 1][0].Raw());
#define BINDP(i, j, k) \
  do { \
    const auto xnor = in##i##j ^ nk##k; \
    const auto l4 = mask4 & xnor; \
    const auto popc_l4 = _mm256_shuffle_epi8(popc_table, l4); \
    const auto h4 = mask4 & _mm256_srli_epi32(xnor, 4); \
    const auto popc_h4 = _mm256_shuffle_epi8(popc_table, h4); \
    const auto cnt = _mm256_add_epi8(popc_l4, popc_h4); \
    xnorsum##i##j##k = _mm256_add_epi8(xnorsum##i##j##k, cnt); \
  } while (0);
                  BINDP(0, 0, 0);
                  BINDP(0, 0, 1);
                  BINDP(0, 1, 0);
                  BINDP(0, 1, 1);
                  in00 = in01;
                  const auto in11 = _mm256_set1_epi32(in_tile[row + kr][col + kc + 1][1].Raw());
                  BINDP(1, 0, 0);
                  BINDP(1, 0, 1);
                  BINDP(1, 1, 0);
                  BINDP(1, 1, 1);
                  in10 = in11;
                }
              }
              const auto psum000 = _mm256_maddubs_epi16(xnorsum000, vone);
              const auto psum001 = _mm256_maddubs_epi16(xnorsum001, vone);
              const auto psum010 = _mm256_maddubs_epi16(xnorsum010, vone);
              const auto psum011 = _mm256_maddubs_epi16(xnorsum011, vone);
              const auto psum100 = _mm256_maddubs_epi16(xnorsum100, vone);
              const auto psum101 = _mm256_maddubs_epi16(xnorsum101, vone);
              const auto psum110 = _mm256_maddubs_epi16(xnorsum110, vone);
              const auto psum111 = _mm256_maddubs_epi16(xnorsum111, vone);
              const auto usum000 = _mm256_hadd_epi16(psum000, psum001);
              const auto usum001 = _mm256_hadd_epi16(psum010, psum011);
              const auto usum010 = _mm256_hadd_epi16(psum100, psum101);
              const auto usum011 = _mm256_hadd_epi16(psum110, psum111);
              const auto usum100 = _mm256_permute4x64_epi64(usum000, 0xD8);
              const auto usum101 = _mm256_permute4x64_epi64(usum001, 0xD8);
              const auto usum110 = _mm256_permute4x64_epi64(usum010, 0xD8);
              const auto usum111 = _mm256_permute4x64_epi64(usum011, 0xD8);
              const auto tmp0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&out_tile[row][col + 0][0]));
              const auto tmp1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&out_tile[row][col + 1][0]));
              const auto nsum = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&notsum[0]));
              const auto diff00 = _mm256_sub_epi16(usum100, nsum);
              const auto diff01 = _mm256_sub_epi16(usum101, nsum);
              const auto diff10 = _mm256_sub_epi16(usum110, nsum);
              const auto diff11 = _mm256_sub_epi16(usum111, nsum);
              const auto shifted00 = _mm256_slli_epi16(diff00, in_bit_ch_high);
              const auto shifted01 = _mm256_slli_epi16(diff01, in_bit_ch_high);
              const auto shifted10 = _mm256_slli_epi16(diff10, in_bit_ch_high + 1);
              const auto shifted11 = _mm256_slli_epi16(diff11, in_bit_ch_high + 1);
              const auto res0 = _mm256_add_epi16(tmp0, _mm256_add_epi16(shifted00, shifted10));
              const auto res1 = _mm256_add_epi16(tmp1, _mm256_add_epi16(shifted01, shifted11));
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_tile[row][col + 0][0]), res0);
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_tile[row][col + 1][0]), res1);
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
  }
  Measurement::Stop();

  using namespace dlk;
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, out_channels, in_height * in_width);

  if (p.thresholds != nullptr) {
    ApplyThresholdsAndPack(output_, p, buf_th.get());
    Measurement::Start("copy");
    std::copy(buf_th.get(), buf_th.get() + out_size / 32 * 2, (QUANTIZED_PACKED*)p.device_output_buf);
    Measurement::Stop();
  } else {
    const T_UINT b = 32;
    Measurement::Start("copy");
    std::copy(p.device_output_buf, p.device_output_buf + out_size, buf_non_th.get());
    Measurement::Stop();
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>::tensor_info_t<std::size_t> buf_shape = {
      out_height, out_width, out_channels
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC> buf_tensor(buf_non_th.get(), buf_shape);
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

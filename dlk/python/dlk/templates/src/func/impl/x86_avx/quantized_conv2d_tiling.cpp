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
  const auto mask4 = _mm256_set1_epi8(0x0F);
  const auto popc_table = _mm256_setr_epi8(
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
  );
  const auto vone = _mm256_set1_epi8(0x01);
  const auto vone_16 = _mm256_set1_epi16(0x0001);
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
            const auto psum0000 = _mm256_maddubs_epi16(xnorsum000, vone);
            const auto psum0001 = _mm256_maddubs_epi16(xnorsum001, vone);
            const auto psum0010 = _mm256_maddubs_epi16(xnorsum010, vone);
            const auto psum0011 = _mm256_maddubs_epi16(xnorsum011, vone);
            const auto psum0100 = _mm256_maddubs_epi16(xnorsum100, vone);
            const auto psum0101 = _mm256_maddubs_epi16(xnorsum101, vone);
            const auto psum0110 = _mm256_maddubs_epi16(xnorsum110, vone);
            const auto psum0111 = _mm256_maddubs_epi16(xnorsum111, vone);
            const auto psum1000 = _mm256_madd_epi16(psum0000, vone_16);
            const auto psum1001 = _mm256_madd_epi16(psum0001, vone_16);
            const auto psum1010 = _mm256_madd_epi16(psum0010, vone_16);
            const auto psum1011 = _mm256_madd_epi16(psum0011, vone_16);
            const auto psum1100 = _mm256_madd_epi16(psum0100, vone_16);
            const auto psum1101 = _mm256_madd_epi16(psum0101, vone_16);
            const auto psum1110 = _mm256_madd_epi16(psum0110, vone_16);
            const auto psum1111 = _mm256_madd_epi16(psum0111, vone_16);
            const auto usum000 = _mm256_packs_epi32(psum1000, psum1001);
            const auto usum001 = _mm256_packs_epi32(psum1010, psum1011);
            const auto usum010 = _mm256_packs_epi32(psum1100, psum1101);
            const auto usum011 = _mm256_packs_epi32(psum1110, psum1111);
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
  Measurement::Stop();

  using namespace dlk;
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, out_channels, in_height * in_width);

  if (p.thresholds != nullptr) {
    ApplyThresholds(output_, p);
    pack_16bit(p.device_output_buf, buf_th.get(), out_size);
    const std::size_t b = 32;
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>::tensor_info_t<std::size_t> buf_shape = {
      out_height, out_width, (out_channels + b - 1) / b, p.n_bit, b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl> buf_tensor(buf_th.get(), buf_shape);
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
    std::copy(p.device_output_buf, p.device_output_buf + out_size, buf_non_th.get());
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>::tensor_info_t<std::size_t> buf_shape = {
      out_height, out_width, out_channels
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC> buf_tensor(buf_non_th.get(), buf_shape);
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl>::tensor_info_t<std::size_t> out_shape = {
      (out_channels + b - 1) / b, out_height, out_width, b
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl> out(p.device_output_buf, out_shape);
    convert_tensor(buf_tensor, out);
  }
}

} // namespace impl

} // namespace dlk

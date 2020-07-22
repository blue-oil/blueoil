/* Copyright 2020 The Blueoil Authors. All Rights Reserved.

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

#include <cstdint>
#include <array>
#include <utility>
#include "global.h"
#include "time_measurement.h"
#include "func/cast.h"

namespace {

uint64_t bitwise_interleave_impl(uint64_t x) {
  uint64_t t = (x ^ (x >> 16)) & UINT64_C(0x00000000FFFF0000);
  x = x ^ t ^ (t << 16);
  t = (x ^ (x >> 8)) & UINT64_C(0x0000FF000000FF00);
  x = x ^ t ^ (t << 8);
  t = (x ^ (x >> 4)) & UINT64_C(0x00F000F000F000F0);
  x = x ^ t ^ (t << 4);
  t = (x ^ (x >> 2)) & UINT64_C(0x0C0C0C0C0C0C0C0C);
  x = x ^ t ^ (t << 2);
  t = (x ^ (x >> 1)) & UINT64_C(0x2222222222222222);
  return x ^ t ^ (t << 1);
}

std::pair<uint64_t, uint64_t> bitwise_interleave(
    uint64_t lsb, uint64_t msb) {
  uint64_t t = (msb ^ (lsb >> 32)) & UINT64_C(0x00000000FFFFFFFF);
  lsb = lsb ^ (t << 32);
  msb = msb ^ t;
  return std::make_pair(bitwise_interleave_impl(lsb), bitwise_interleave_impl(msb));
}

uint32_t bitwise_interleave_impl(uint32_t x) {
  uint32_t t = (x ^ (x >> 8)) & 0x0000FF00;
  x = x ^ t ^ (t << 8);
  t = (x ^ (x >> 4)) & 0x00F000F0;
  x = x ^ t ^ (t << 4);
  t = (x ^ (x >> 2)) & 0x0C0C0C0C;
  x = x ^ t ^ (t << 2);
  t = (x ^ (x >> 1)) & 0x22222222;
  return x ^ t ^ (t << 1);
}

std::pair<uint32_t, uint32_t> bitwise_interleave(
    uint32_t lsb, uint32_t msb) {
  uint32_t t = (msb ^ (lsb >> 16)) & 0x0000FFFF;
  lsb = lsb ^ (t << 16);
  msb = msb ^ t;
  return std::make_pair(bitwise_interleave_impl(lsb), bitwise_interleave_impl(msb));
}

uint16_t bitwise_interleave_impl(uint16_t x) {
  uint16_t t = (x ^ (x >> 4)) & 0x00F0;
  x = x ^ t ^ (t << 4);
  t = (x ^ (x >> 2)) & 0x0C0C;
  x = x ^ t ^ (t << 2);
  t = (x ^ (x >> 1)) & 0x2222;
  return x ^ t ^ (t << 1);
}

std::pair<uint16_t, uint16_t> bitwise_interleave(
    uint16_t lsb, uint16_t msb) {
  uint16_t t = (msb ^ (lsb >> 8)) & 0x00FF;
  lsb = lsb ^ (t << 8);
  msb = msb ^ t;
  return std::make_pair(bitwise_interleave_impl(lsb), bitwise_interleave_impl(msb));
}

uint8_t bitwise_interleave_impl(uint8_t x) {
  uint8_t t = (x ^ (x >> 2)) & 0x0C;
  x = x ^ t ^ (t << 2);
  t = (x ^ (x >> 1)) & 0x22;
  return x ^ t ^ (t << 1);
}

std::pair<uint8_t, uint8_t> bitwise_interleave(
    uint8_t lsb, uint8_t msb) {
  uint8_t t = (msb ^ (lsb >> 4)) & 0x0F;
  lsb = lsb ^ (t << 4);
  msb = msb ^ t;
  return std::make_pair(bitwise_interleave_impl(lsb), bitwise_interleave_impl(msb));
}

void linear_to_float(
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& input,
    const T_FLOAT max_value,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  Measurement::Start("linear_to_float");

  constexpr std::size_t b = 32;
  constexpr std::size_t n_bit = 2;
  constexpr std::size_t table_size = 1 << n_bit;
  const auto height = output.get_shape()[1];
  const auto width = output.get_shape()[2];
  const auto true_out_channels = output.get_shape()[3];
  const auto channel_blocks = (true_out_channels + b - 1) / b;
  const auto out_channels = channel_blocks * b;

  std::array<T_FLOAT, table_size> table;
  const T_FLOAT n = (1 << n_bit) - 1;
  const T_FLOAT coeff = max_value / n;
  for (size_t i = 0; i < table_size; ++i) {
    table[i] = i * coeff;
  }

  constexpr auto half_b = b / 2;
  auto out_buf = input.data();
  const auto area = height * width;
  const auto blocks = channel_blocks * area;
#pragma omp parallel for
  for (size_t i = 0; i < blocks; ++i) {
    const auto ch = i / area;
    const auto pos = i % area;
    const auto rem = true_out_channels - ch*b;
    const QUANTIZED_PACKED::base_t lsb = out_buf[i * n_bit].Raw();
    const QUANTIZED_PACKED::base_t msb = out_buf[i * n_bit + 1].Raw();
    auto interleaved = bitwise_interleave(lsb, msb);
    if (rem >= b) {
      for (size_t d = 0; d < half_b; ++d) {
        auto bits = (interleaved.first >> (2 * d)) & 0b11;
        size_t out_idx = pos * true_out_channels + ch * b + d;
        output.data()[out_idx] = table[bits];
      }
      for (size_t d = half_b; d < b; ++d) {
        auto bits = (interleaved.second >> (2 * d - b)) & 0b11;
        size_t out_idx = pos * true_out_channels + ch * b + d;
        output.data()[out_idx] = table[bits];
      }
    } else {
      for (size_t d = 0; d < half_b; ++d) {
        if (d >= rem) break;
        auto bits = (interleaved.first >> (2 * d)) & 0b11;
        size_t out_idx = pos * true_out_channels + ch * b + d;
        output.data()[out_idx] = table[bits];
      }
      for (size_t d = half_b; d < b; ++d) {
        if (d >= rem) break;
        auto bits = (interleaved.second >> (2 * d - b)) & 0b11;
        size_t out_idx = pos * true_out_channels + ch * b + d;
        output.data()[out_idx] = table[bits];
      }
    }
  }
  Measurement::Stop();
}

} // namespace

void func_Cast(const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& input,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  Measurement::Start("Cast");

  constexpr T_FLOAT max_value = 2.0;
  linear_to_float(input, max_value, output);

  Measurement::Stop();
}

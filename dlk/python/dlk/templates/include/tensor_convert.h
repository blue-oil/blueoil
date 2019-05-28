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

#ifndef DLK_TENSOR_CONVERT_H_INCLUDED
#define DLK_TENSOR_CONVERT_H_INCLUDED

#include "global.h"
#include "tensor_view.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "func/impl/quantized_conv2d_tiling.h"
#include "func/impl/quantized_conv2d_dim2col.h"

inline void convert_tensor(const TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>& before,
    const TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl>& after) {
  const auto in_shape = before.get_shape();
  const auto in_height = in_shape[0];
  const auto in_width = in_shape[1];
  const auto out_shape = after.get_shape();
  const auto channel_high = out_shape[0];
  const auto channel_low = out_shape[3];
  for (std::size_t dh = 0; dh < channel_high; ++dh)
    for (std::size_t r = 0; r < in_height; ++r)
      for (std::size_t c = 0; c < in_width; ++c)
        for (std::size_t dl = 0; dl < channel_low; ++dl)
          after(dh, r, c, dl) = before(r, c, dh * channel_low + dl);
}

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& before,
    const dlk::impl::dim2col_input_t& after,
    const binary_convolution_parameters& p) {
  const auto& np = p.normal_conv_params;
  const auto in_shape = before.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto in_channel = in_shape[0];
  const auto bits = in_shape[3];
  const auto out_height = np.output_height;
  const auto out_width = np.output_width;
  const auto kh = np.kernel_height;
  const auto kw = np.kernel_width;
  const auto pad = np.padding;
  for (T_INT i = 0; i < out_height; ++i)
    for (T_INT j = 0; j < out_width; ++j)
      for (T_INT k = 0; k < in_channel; ++k)
        for (T_INT d = 0; d < bits; ++d)
          for (T_INT kr = 0; kr < kh; ++kr)
            for (T_INT kc = 0; kc < kw; ++kc) {
              const auto r = i + kr - pad;
              const auto c = j + kc - pad;
              if (r >= 0 && r < in_height && c >= 0 && c < in_width) {
                after(i * out_width + j, k, d, 0) = before(k, r, c, d, 0);
              } else {
                after(i * out_width + j, k, d, 0) = QUANTIZED_PACKED(0);
              }
            }
}

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& before,
    const dlk::impl::dim2col_input_t& after,
    const binary_convolution_parameters& p) {
  const auto& np = p.normal_conv_params;
  const auto in_shape = before.get_shape();
  const auto in_height = in_shape[0];
  const auto in_width = in_shape[1];
  const auto in_channel = in_shape[2];
  const auto bits = in_shape[3];
  const auto out_height = np.output_height;
  const auto out_width = np.output_width;
  const auto kh = np.kernel_height;
  const auto kw = np.kernel_width;
  const auto pad = np.padding;
  for (T_INT i = 0; i < out_height; ++i)
    for (T_INT j = 0; j < out_width; ++j)
      for (T_INT k = 0; k < in_channel; ++k)
        for (T_INT d = 0; d < bits; ++d)
          for (T_INT kr = 0; kr < kh; ++kr)
            for (T_INT kc = 0; kc < kw; ++kc) {
              const auto r = i + kr - pad;
              const auto c = j + kc - pad;
              if (r >= 0 && r < in_height && c >= 0 && c < in_width) {
                after(i * out_width + j, k, d, 0) = before(r, c, k, d, 0);
              } else {
                after(i * out_width + j, k, d, 0) = QUANTIZED_PACKED(0);
              }
            }
}

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& before,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& after) {
  const auto in_shape = before.get_shape();
  const auto height = in_shape[0];
  const auto width = in_shape[1];
  const auto channel = in_shape[2];
  const auto bits = in_shape[3];
  for (std::size_t i = 0; i < height; ++i)
    for (std::size_t j = 0; j < width; ++j)
      for (std::size_t k = 0; k < channel; ++k)
        for (std::size_t d = 0; d < bits; ++d)
          after(k, i, j, d, 0) = before(i, j, k, d, 0);
}

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& before,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& after) {
  const auto in_shape = before.get_shape();
  const auto height = in_shape[1];
  const auto width = in_shape[2];
  const auto channel = in_shape[0];
  const auto bits = in_shape[3];
  for (std::size_t i = 0; i < height; ++i)
    for (std::size_t j = 0; j < width; ++j)
      for (std::size_t k = 0; k < channel; ++k)
        for (std::size_t d = 0; d < bits; ++d)
          after(i, j, k, d, 0) = before(k, i, j, d, 0);
}

inline void convert_tensor(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& before,
    const dlk::impl::tiling_input_t& after) {
  dlk::impl::pack_input_for_tiling(before, after);
}

inline void convert_tensor(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& before,
    const dlk::impl::dim2col_input_t& after,
    const binary_convolution_parameters& p) {
  dlk::impl::im2col(before, after, p);
}

template <typename T, MemoryLayout layout>
void convert_tensor(const TensorView<T, layout>& before,
    const TensorView<T, layout>& after) {
  auto num_elems = before.size();
  std::copy(before.data(), before.data() + num_elems, after.data());
}

#endif

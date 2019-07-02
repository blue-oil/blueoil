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

#ifndef DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED
#define DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED

#include <vector>
#include <memory>
#include <stdexcept>

#include "tensor_view.h"
#include "tensor_convert.h"
#include "operators.h"
#include "time_measurement.h"
#include "func/impl/apply_thresholds.h"
#include "func/impl/quantized_conv2d_tiling.h"

template <typename T, MemoryLayout layout>
void QuantizedConv2D(const TensorView<T, layout>& input,
    const kernel_t& kernel,
    binary_convolution_parameters p) {
  Measurement::Start("QuantizedConv2D");

  constexpr T_UINT TilingInTypeBitWidth = dlk::impl::tiling_input_elem_t::BitCount;
  T_UINT kh = p.normal_conv_params.kernel_height;
  T_UINT kw = p.normal_conv_params.kernel_width;
  T_UINT padding = p.normal_conv_params.padding;
  T_UINT ih = p.normal_conv_params.input_height;
  T_UINT iw = p.normal_conv_params.input_width;
  T_UINT ic = p.normal_conv_params.kernel_depth;
  T_UINT oc = p.normal_conv_params.output_channels;
  auto size = oc * ih * iw;
  if (p.device_output_buf == nullptr)
    p.device_output_buf = new BIN_CONV_OUTPUT[size]();

  if ((kh == 3 && kw == 3 && padding == 1) ||
      (kh == 1 && kw == 1 && padding == 0)) {
#ifdef RUN_ON_FPGA
    dlk::impl::kn2row_input_t::tensor_info_t<std::size_t> shape = {
      (ic + QUANTIZED_PACKED::BitCount - 1) / QUANTIZED_PACKED::BitCount,
      ih,
      iw,
      p.bin_input_bitwidth,
      QUANTIZED_PACKED::BitCount
    };
    dlk::impl::kn2row_input_t tmp(p.device_input_buf, shape);
    Measurement::Start("Tensor convert");
    convert_tensor(input, tmp);
    Measurement::Stop();
    dlk::impl::TCAConv2d(tmp, kernel, p);
#elif defined USE_NEON
    dlk::impl::tiling_input_t::tensor_info_t<std::size_t> shape = {
      ic / TilingInTypeBitWidth,
      ih,
      iw,
      p.bin_input_bitwidth,
      TilingInTypeBitWidth
    };
    dlk::impl::tiling_input_t tmp(p.device_input_buf, shape);
    Measurement::Start("Tensor convert");
    convert_tensor(input, tmp);
    Measurement::Stop();
    dlk::impl::QuantizedConv2DTiling(tmp, kernel, p);
#else
    dlk::impl::kn2row_input_t::tensor_info_t<std::size_t> shape = {
      ih,
      iw,
      ic / QUANTIZED_PACKED::BitCount,
      p.bin_input_bitwidth,
      QUANTIZED_PACKED::BitCount
    };
    dlk::impl::kn2row_input_t tmp(p.device_input_buf, shape);
    Measurement::Start("Tensor convert");
    convert_tensor(input, tmp);
    Measurement::Stop();
    dlk::impl::QuantizedConv2DKn2Row(tmp, kernel, p);
#endif
  } else {
    throw std::invalid_argument("Unsupported convolution parameter");
  }

  Measurement::Stop();
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2D(
    const TensorView<T, layout>& input,
    const kernel_t& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  QuantizedConv2D(input, kernel, p);

  Measurement::Start("QuantizedConv2D_ApplyScalingFactor");

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  // temporary: (2^n - 1) * (max - min)
  const T_FLOAT post_qtz_factor = 2.0f / 3.0f;

  int b = 32;
  auto &ncp(p.normal_conv_params);
  auto true_out_channels = output.get_shape()[3];
  auto channel_blocks = (true_out_channels + b - 1) / b;

  int out_index = 0;
  for (int h = 0; h < ncp.output_height; ++h)
    for (int w = 0; w < ncp.output_width; ++w)
      for (int s = 0; s < channel_blocks; ++s)
        for (int d = 0; d < std::min(b, (int)true_out_channels - s*b); ++d)
          output.data()[out_index++] = (scaling_factor * post_qtz_factor) * p.device_output_buf[h * (b * ncp.output_width) + w * b + s * (ncp.output_height * ncp.output_width * b) + d];

  Measurement::Stop();

}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2D(
    const TensorView<T, layout>& input,
    const kernel_t& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    T_FLOAT scaling_factor[],
    binary_convolution_parameters p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems =
      p.normal_conv_params.output_height * p.normal_conv_params.output_width;
  unsigned out_channels = p.normal_conv_params.output_channels;

  int b = 32;
  auto& ncp(p.normal_conv_params);
  auto true_out_channels = output.get_shape()[3];
  auto channel_blocks = (true_out_channels + b - 1) / b;

  // temporary: (2^n - 1) * (max - min)
  T_FLOAT post_qtz_factor = 2.0 / 3.0;

  Measurement::Start("QuantizedConv2D_ApplyScalingFactor");

  int out_index = 0;
  for (int h = 0; h < ncp.output_height; ++h)
    for (int w = 0; w < ncp.output_width; ++w)
      for (int s = 0; s < channel_blocks; ++s)
        for (int d = 0; d < std::min(b, (int)true_out_channels - s*b); ++d)
          output.data()[out_index++] = (scaling_factor[s*b + d] * post_qtz_factor) * p.device_output_buf[h * (b * ncp.output_width) + w * b + s * (ncp.output_height * ncp.output_width * b) + d];

  Measurement::Stop();
}

template<typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const kernel_t& kernel,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  const auto bytes = out_elems / 8 * p.n_bit;

  Measurement::Start("Memcpy");

  memcpy(output.data(), (void*)p.device_output_buf, bytes);

  Measurement::Stop();
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const kernel_t& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  QuantizedConv2D(input, kernel, p);

  Measurement::Start("linear_to_float");

  T_FLOAT n = (1 << p.n_bit) - 1;
  const auto& np = p.normal_conv_params;
  const auto out_height = np.output_height;
  const auto out_width = np.output_width;
  const auto out_channels = np.output_channels;
  const auto true_out_channels = output.get_shape()[3];

  QUANTIZED_PACKED::base_t* ptr = (QUANTIZED_PACKED::base_t*)p.device_output_buf;
  for (unsigned r = 0; r < out_height; ++r) {
    for (unsigned c = 0; c < out_width; ++c) {
      for (unsigned d = 0; d < true_out_channels; ++d) {
        const auto i = r * out_width * p.n_bit + c * p.n_bit;
        QUANTIZED_PACKED::base_t bits = 0;
        for (unsigned digit = 0; digit < p.n_bit; ++digit) {
          bits |= ((ptr[i + digit] >> d) & 1) << digit;
        }
        T_FLOAT tmp = (T_FLOAT)bits;
        tmp = tmp / n;
        output(0, r, c, d) = tmp * p.max_value;
      }
    }
  }

  Measurement::Stop();
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const kernel_t& kernel,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    const T_FLOAT scaling_factor[],
    const binary_convolution_parameters& p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const kernel_t& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    T_FLOAT scaling_factor[],
    binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

#endif // DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED



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
#include "func/impl/quantized_conv2d_dim2col.h"
#include "func/impl/quantized_conv2d_tiling.h"
#include "func/impl/quantized_conv2d_kn2row.h"

inline void func_linear_to_float(
    const BIN_CONV_OUTPUT input[],
    T_INT nbit,
    T_FLOAT max_value,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  T_FLOAT n = (1 << nbit) - 1;
  unsigned num_elems = output.size();

  for (unsigned i = 0; i < num_elems; i++) {
    T_FLOAT tmp = (T_FLOAT)input[i];
    tmp = tmp / n;
    output.data()[i] = tmp * max_value;
  }
}

template <typename T, MemoryLayout layout>
void QuantizedConv2D(const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    binary_convolution_parameters p) {
  constexpr T_UINT TilingInTypeBitWidth = dlk::impl::tiling_input_elem_t::BitCount;
  int kh = p.normal_conv_params.kernel_height;
  int kw = p.normal_conv_params.kernel_width;
  int padding = p.normal_conv_params.padding;
  int ih = p.normal_conv_params.input_height;
  int iw = p.normal_conv_params.input_width;
  int ic = p.normal_conv_params.kernel_depth;
  int oc = p.normal_conv_params.output_channels;
  auto size = oc * ih * iw;
  if (p.device_output_buf == nullptr)
    p.device_output_buf = new BIN_CONV_OUTPUT[size]();
  else
    std::memset((void *)p.device_output_buf, 0, size * sizeof(BIN_CONV_OUTPUT));

#ifdef RUN_ON_FPGA
  if ((kh == 3 && kw == 3 && padding == 1) ||
      (kh == 1 && kw == 1 && padding == 0)) {
    dlk::impl::kn2row_input_t::tensor_info_t<std::size_t> shape = {
      (ic + QUANTIZED_PACKED::BitCount - 1) / QUANTIZED_PACKED::BitCount,
      ih,
      iw,
      p.bin_input_bitwidth,
      QUANTIZED_PACKED::BitCount
    };
    dlk::impl::kn2row_input_t tmp(p.device_input_buf, shape);
    convert_tensor(input, tmp);
    dlk::impl::TCAConv2d(tmp, kernel, p);
  } else {
    throw std::invalid_argument("Unsupported convolution parameter");
  }
#else
  if ((kh == 3 && kw == 3 && padding == 1) ||
      (kh == 1 && kw == 1 && padding == 0)) {
#if defined(USE_NEON)
    if ((ic % TilingInTypeBitWidth) == 0) {
      const auto kernel_buf_size = kh * kw * ic * oc / 32;
      const auto kernel_ohwc_raw = std::make_unique<QUANTIZED_PACKED_KERNEL[]>(kernel_buf_size);
      kernel_t::tensor_info_t<std::size_t> kernel_shape = {
        oc, kh, kw, ic
      };
      kernel_t kernel_ohwc(kernel_ohwc_raw.get(), kernel_shape);
      convert_tensor(kernel, kernel_ohwc);
      dlk::impl::tiling_input_t::tensor_info_t<std::size_t> shape = {
        ic / TilingInTypeBitWidth,
        ih,
        iw,
        p.bin_input_bitwidth,
        TilingInTypeBitWidth
      };
      dlk::impl::tiling_input_t tmp(p.device_input_buf, shape);
      convert_tensor(input, tmp);
      dlk::impl::QuantizedConv2DTiling(tmp, kernel_ohwc, p);
    } else {
#endif
      const auto kernel_buf_size = kh * kw * ic * oc / 32;
      const auto kernel_hwoi_raw = std::make_unique<QUANTIZED_PACKED_KERNEL[]>(kernel_buf_size);
      dlk::impl::kn2row_kernel_t::tensor_info_t<std::size_t> kernel_shape = {
        kh, kw, oc, ic
      };
      dlk::impl::kn2row_kernel_t kernel_hwoi(kernel_hwoi_raw.get(), kernel_shape);
      convert_tensor(kernel, kernel_hwoi);
      dlk::impl::kn2row_input_t::tensor_info_t<std::size_t> shape = {
        (ic + QUANTIZED_PACKED::BitCount - 1) / QUANTIZED_PACKED::BitCount,
        ih,
        iw,
        p.bin_input_bitwidth,
        QUANTIZED_PACKED::BitCount
      };
      dlk::impl::kn2row_input_t tmp(p.device_input_buf, shape);
      convert_tensor(input, tmp);
      dlk::impl::QuantizedConv2DKn2Row(tmp, kernel_hwoi, p);
#ifdef USE_NEON
    }
#endif
  } else {
    const auto kernel_buf_size = kh * kw * ic * oc / 32;
    const auto kernel_ohwc_raw = std::make_unique<QUANTIZED_PACKED_KERNEL[]>(kernel_buf_size);
    kernel_t::tensor_info_t<std::size_t> kernel_shape = {
      oc, kh, kw, ic
    };
    kernel_t kernel_ohwc(kernel_ohwc_raw.get(), kernel_shape);
    convert_tensor(kernel, kernel_ohwc);
    dlk::impl::dim2col_input_t::tensor_info_t<std::size_t> shape = {
      kh * kw * ic,
      ih * iw
    };
    dlk::impl::dim2col_input_t tmp(p.device_input_buf, shape);
    convert_tensor(input, tmp, p);
    dlk::impl::QuantizedConv2DIm2Col(tmp, kernel_ohwc, p);
  }
#endif
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2D(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  Measurement::Start("QuantizedConv2D");

  QuantizedConv2D(input, kernel, p);

  Measurement::Stop();

  Measurement::Start("QuantizedConv2D_ApplyScalingFactor");

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  // temporary: (2^n - 1) * (max - min)
  const T_FLOAT post_qtz_factor = 2.0f / 3.0f;

  int b = 32;
  auto &ncp(p.normal_conv_params);

  if (ncp.output_channels > b) {
    int out_index = 0;
    for (int h = 0; h < ncp.output_height; ++h)
      for (int w = 0; w < ncp.output_width; ++w)
        for (int s = 0; s < ncp.output_channels / b; ++s)
          for (int d = 0; d < b; ++d)
            output.data()[out_index++] = (scaling_factor * post_qtz_factor) * p.device_output_buf[h * (b * ncp.output_width) + w * b + s * (ncp.output_height * ncp.output_width * b) + d];
  } else {
    int tca_channels = ((ncp.output_channels + b - 1) / b) * b;
    int out_index = 0;
    for (int h = 0; h < ncp.output_height; ++h)
      for (int w = 0; w < ncp.output_width; ++w)
        for (int d = 0; d < ncp.output_channels; ++d)
          output.data()[out_index++] = (scaling_factor * post_qtz_factor) * p.device_output_buf[h * (tca_channels * ncp.output_width) + w * tca_channels + d];
  }

  Measurement::Stop();

}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2D(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    T_FLOAT scaling_factor[],
    binary_convolution_parameters p) {
  Measurement::Start("QuantizedConv2D");

  QuantizedConv2D(input, kernel, p);

  Measurement::Stop();

  unsigned out_elems =
      p.normal_conv_params.output_height * p.normal_conv_params.output_width;
  unsigned out_channels = p.normal_conv_params.output_channels;

  int b = 32;
  auto& ncp(p.normal_conv_params);
  int tca_channels = ((ncp.output_channels + b - 1) / b) * b;

  // temporary: (2^n - 1) * (max - min)
  T_FLOAT post_qtz_factor = 2.0 / 3.0;

  if (ncp.output_channels > b) {
    Measurement::Start("QuantizedConv2D_ChangeOutputLayout");
    int out_index = 0;
    for (int h = 0; h < ncp.output_height; ++h)
      for (int w = 0; w < ncp.output_width; ++w)
        for (int s = 0; s < ncp.output_channels / b; ++s)
          for (int d = 0; d < b; ++d)
            output.data()[out_index++] = p.device_output_buf[h * (b * ncp.output_width) + w * b + s * (ncp.output_height * ncp.output_width * b) + d];
    Measurement::Stop();

    Measurement::Start("QuantizedConv2D_ApplyScalingFactor");
    for (int i = 0; i < out_elems; ++i)
      for (int c = 0; c < out_channels; ++c)
        output.data()[i * out_channels + c] *= (scaling_factor[c] * post_qtz_factor);
    Measurement::Stop();
  } else {
    Measurement::Start("QuantizedConv2D_RemoveChannels");
    int tmp_index = 0;
    auto tmp_output = std::make_unique<T_FLOAT[]>(out_elems * ncp.output_channels);
    for (int h = 0; h < ncp.output_height; ++h)
      for (int w = 0; w < ncp.output_width; ++w)
        for (int d = 0; d < ncp.output_channels; ++d)
          tmp_output[tmp_index++] = p.device_output_buf[h * (tca_channels * ncp.output_width) + w * tca_channels + d];
    Measurement::Stop();

    Measurement::Start("QuantizedConv2D_ApplyScalingFactor");
    int out_index = 0;
    for (int h = 0; h < ncp.output_height; ++h)
      for (int w = 0; w < ncp.output_width; ++w)
        for (int d = 0; d < ncp.output_channels; ++d)
          output.data()[out_index++] = (scaling_factor[d] * post_qtz_factor) * p.device_output_buf[h * (tca_channels * ncp.output_width) + w * tca_channels + d];
    Measurement::Stop();
  }
}

template<typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  const auto bytes = out_elems / 8 * p.n_bit;
  memcpy(output.data(), (void*)p.device_output_buf, bytes);
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  func_linear_to_float(p.device_output_buf, p.n_bit, p.max_value, output);
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    const T_FLOAT scaling_factor[],
    const binary_convolution_parameters& p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::OhIhHWOlIl>& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    T_FLOAT scaling_factor[],
    binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

#endif // DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED



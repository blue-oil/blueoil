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
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
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

  if ((kh == 3 && kw == 3 && padding == 1) ||
      (kh == 1 && kw == 1 && padding == 0)) {
    if ((ic % TilingInTypeBitWidth) == 0) {
#if defined(USE_NEON) && !defined(RUN_ON_FPGA)
      dlk::impl::tiling_input_t::tensor_info_t<std::size_t> shape = {
        ic / TilingInTypeBitWidth,
        ih,
        iw,
        p.bin_input_bitwidth,
        TilingInTypeBitWidth
      };
      dlk::impl::tiling_input_t tmp(p.device_input_buf, shape);
      convert_tensor(input, tmp);
      dlk::impl::QuantizedConv2DTiling(tmp, kernel, p);
#else
#ifndef RUN_ON_FPGA
      const auto kernel_buf_size = kh * kw * ic * oc / 32;
      const auto kernel_hwoi_raw = std::make_unique<QUANTIZED_PACKED_KERNEL[]>(kernel_buf_size);
      dlk::impl::kn2row_kernel_t::tensor_info_t<std::size_t> kernel_shape = {
        kh, kw, oc, ic
      };
      dlk::impl::kn2row_kernel_t kernel_hwoi(kernel_hwoi_raw.get(), kernel_shape);
      convert_tensor(kernel, kernel_hwoi, p);
#endif
      dlk::impl::kn2row_input_t::tensor_info_t<std::size_t> shape = {
        ih,
        iw,
        (ic + QUANTIZED_PACKED::BitCount - 1) / QUANTIZED_PACKED::BitCount,
        p.bin_input_bitwidth,
        QUANTIZED_PACKED::BitCount
      };
      dlk::impl::kn2row_input_t tmp(p.device_input_buf, shape);
      convert_tensor(input, tmp);
#ifdef RUN_ON_FPGA
      dlk::impl::TCAConv2d(tmp, kernel, p);
#else
      dlk::impl::QuantizedConv2DKn2Row(tmp, kernel_hwoi, p);
#endif
#endif
    } else {
#ifndef RUN_ON_FPGA
      const auto kernel_buf_size = kh * kw * ic * oc / 32;
      const auto kernel_hwoi_raw = std::make_unique<QUANTIZED_PACKED_KERNEL[]>(kernel_buf_size);
      dlk::impl::kn2row_kernel_t::tensor_info_t<std::size_t> kernel_shape = {
        kh, kw, oc, ic
      };
      dlk::impl::kn2row_kernel_t kernel_hwoi(kernel_hwoi_raw.get(), kernel_shape);
      convert_tensor(kernel, kernel_hwoi, p);
#endif
      dlk::impl::kn2row_input_t::tensor_info_t<std::size_t> shape = {
        ih,
        iw,
        (ic + QUANTIZED_PACKED::BitCount - 1) / QUANTIZED_PACKED::BitCount,
        p.bin_input_bitwidth,
        QUANTIZED_PACKED::BitCount
      };
      dlk::impl::kn2row_input_t tmp(p.device_input_buf, shape);
      convert_tensor(input, tmp);
#ifdef RUN_ON_FPGA
      dlk::impl::TCAConv2d(tmp, kernel, p);
#else
      dlk::impl::QuantizedConv2DKn2Row(tmp, kernel_hwoi, p);
#endif
    }
  } else {
    dlk::impl::dim2col_input_t::tensor_info_t<std::size_t> shape = {
      kh * kw * ic,
      ih * iw
    };
    dlk::impl::dim2col_input_t tmp(p.device_input_buf, shape);
    convert_tensor(input, tmp, p);
    dlk::impl::QuantizedConv2DIm2Col(tmp, kernel, p);
  }
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2D(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
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

  for (unsigned i = 0; i < out_elems; ++i) {
    output.data()[i] = (scaling_factor * post_qtz_factor) * p.device_output_buf[i];
  }

  Measurement::Stop();

}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2D(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    T_FLOAT scaling_factor[],
    binary_convolution_parameters p) {
  Measurement::Start("QuantizedConv2D");

  QuantizedConv2D(input, kernel, p);

  Measurement::Stop();

  Measurement::Start("QuantizedConv2D_ApplyScalingFactor");

  unsigned out_elems =
      p.normal_conv_params.output_height * p.normal_conv_params.output_width;
  unsigned out_channels = p.normal_conv_params.output_channels;

  // temporary: (2^n - 1) * (max - min)
  T_FLOAT post_qtz_factor = 2.0 / 3.0;

  for (unsigned i = 0; i < out_elems; ++i) {
    for (unsigned c = 0; c < out_channels; c++) {
      unsigned idx = i * out_channels + c;
      BIN_CONV_OUTPUT out = p.device_output_buf[idx];
      output.data()[idx] =
          (scaling_factor[c] * post_qtz_factor) * static_cast<T_FLOAT>(out);
    }
  }

  Measurement::Stop();
}

template<typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    const T_FLOAT scaling_factor,
    const binary_convolution_parameters& p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  const auto bytes = out_elems / 8 * p.n_bit;
  memcpy(output.data(), p.device_output_buf, bytes);
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
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
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    const T_FLOAT scaling_factor[],
    const binary_convolution_parameters& p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

template <typename T, MemoryLayout layout>
void func_QuantizedConv2DWithThreshold(
    const TensorView<T, layout>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& kernel,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    T_FLOAT scaling_factor[],
    binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

#endif // DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED



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
#include <cstring>
#include <algorithm>
#include <limits>

#include "global.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "func/impl/apply_thresholds.h"
#include "matrix_view.h"
#include "matrix/quantized_multiplication.h"
#include "matrix/shift_add.h"
#include "time_measurement.h"
#include "tensor_convert.h"
#include "func/impl/pack_16bit.h"

namespace dlk {

namespace impl {

void convert_thresholds(BIN_CONV_OUTPUT *input, BIN_CONV_OUTPUT *output, std::size_t channels) {
  std::memcpy(output, input, channels * NUM_OF_A2W1_THRESHOLD * sizeof(BIN_CONV_OUTPUT));
}

void QuantizedConv2DKn2Row(const kn2row_input_t& input,
    const kn2row_kernel_t& kernel,
    const binary_convolution_parameters &p) {
  using namespace dlk;

  T_UINT ic = p.normal_conv_params.kernel_depth;
  T_UINT ih = p.normal_conv_params.input_height;
  T_UINT iw = p.normal_conv_params.input_width;
  T_UINT oc = p.normal_conv_params.output_channels;
  T_UINT oh = p.normal_conv_params.output_height;
  T_UINT ow = p.normal_conv_params.output_width;
  T_UINT kh = p.normal_conv_params.kernel_height;
  T_UINT kw = p.normal_conv_params.kernel_width;
  T_UINT maxa = (1 << p.n_bit) - 1;
  BYTE *temp_buf_ptr = p.normal_conv_params.temporary_buf;

  assert(ih * iw == oh * ow);

  Measurement::Start("quantized-kn2row");

  auto out_buf = reinterpret_cast<BIN_CONV_OUTPUT*>(p.device_output_buf);
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      out_buf, oc, ih * iw);
  auto kernel_ = MatrixView<QUANTIZED_PACKED_KERNEL, MatrixOrder::RowMajor>(
      kernel.data(), oc * kh * kw, ic / 32);

  assert(kh == kw);
  assert(kh % 2 == 1);
  assert(1 <= kh && kh <= 5);
  assert(ic * kh * kw * maxa <= std::numeric_limits<BIN_CONV_OUTPUT>::max());

  if (kh >= 3) {
    std::fill(out_buf, out_buf + oc * oh * ow, 0);
    for (std::size_t offset = 0; offset < ih * iw; offset += MAX_SIZE_KN2ROW_COL_BLOCK) {
      const auto col_block = std::min(static_cast<std::size_t>(MAX_SIZE_KN2ROW_COL_BLOCK), ih * iw - offset);
      auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
          input.data() + offset * ic / 16, ic / 16, col_block);
      auto buf_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
          reinterpret_cast<BIN_CONV_OUTPUT*>(temp_buf_ptr), oc * kh * kw, col_block);

      quantized_matrix_multiplication(kernel_, input_, buf_);
      matrix_shift_add(buf_, output_, p.normal_conv_params, offset);
    }
  } else {
    auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
        input.data(), ic / 16, ih * iw);
    auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
        out_buf, oc, ih * iw);
    quantized_matrix_multiplication(kernel_, input_, output_);
  }

  const auto out_size = oc * oh * ow;
  if (p.thresholds != nullptr) {
    QUANTIZED_PACKED *buf_ptr = reinterpret_cast<QUANTIZED_PACKED*>(temp_buf_ptr);
    ApplyThresholds(output_, p);
    pack_16bit(out_buf, buf_ptr, out_size);
    const std::size_t b = 32;
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>::tensor_info_t<std::size_t> buf_shape = {
      oh, ow, (oc + b - 1) / b, p.n_bit, b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl> buf_tensor(buf_ptr, buf_shape);
    TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>::tensor_info_t<std::size_t> out_shape = {
      (oc + b - 1) / b,
      oh,
      ow,
      p.n_bit,
      b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl> out(reinterpret_cast<QUANTIZED_PACKED*>(p.device_output_buf), out_shape);
    convert_tensor(buf_tensor, out);
  } else {
    BIN_CONV_OUTPUT *buf_ptr = reinterpret_cast<BIN_CONV_OUTPUT*>(temp_buf_ptr);
    const std::size_t b = 32;
    std::copy(out_buf, out_buf + out_size, buf_ptr);
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>::tensor_info_t<std::size_t> buf_shape = {
      oh, ow, oc 
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC> buf_tensor(buf_ptr, buf_shape);
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl>::tensor_info_t<std::size_t> out_shape = {
      (oc + b - 1) / b, oh, ow, b
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl> out(reinterpret_cast<BIN_CONV_OUTPUT*>(p.device_output_buf), out_shape);
    convert_tensor(buf_tensor, out);
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

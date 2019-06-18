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

void QuantizedConv2DKn2Row(const kn2row_input_t& input,
                                  const kernel_t& kernel,
                                  const binary_convolution_parameters &p) {
  using namespace dlk;

  int ic = p.normal_conv_params.kernel_depth;
  int ih = p.normal_conv_params.input_height;
  int iw = p.normal_conv_params.input_width;
  int oc = p.normal_conv_params.output_channels;
  int oh = p.normal_conv_params.output_height;
  int ow = p.normal_conv_params.output_width;
  int kh = p.normal_conv_params.kernel_height;
  int kw = p.normal_conv_params.kernel_width;

  assert(ih * iw == oh * ow);
  assert(MAX_SIZE_IM2COL_INPUTS_PER_LAYER >= ic * kh * kw * ih * iw);

  Measurement::Start("quantized-kn2row");

  auto kernel_ = MatrixView<QUANTIZED_PACKED_KERNEL, MatrixOrder::RowMajor>(
      kernel.data(), oc * kh * kw, ic / 32);
  auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
      input.data(), ic / 16, ih * iw);
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, oc, ih * iw);

  if (kh == kw && kw == 3) {
    unsigned bufsize = oc * kh * kw * ih * iw;
    BIN_CONV_OUTPUT *kn2row_buf = new BIN_CONV_OUTPUT[bufsize]();
    auto buf_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
        kn2row_buf, oc * kh * kw, ih * iw);

    quantized_matrix_multiplication(kernel_, input_, buf_);
    std::fill(p.device_output_buf, p.device_output_buf + oc * oh * ow, 0);
    matrix_shift_add(buf_, output_, p.normal_conv_params);
    delete[] kn2row_buf;
  } else if (kh == kw && kw == 1) {
    quantized_matrix_multiplication(kernel_, input_, output_);
  } else {
    std::cerr << "Only 1x1 or 3x3 convolutions are supported." << std::endl;
    assert(false);
  }

  const auto out_size = oc * oh * ow;
  if (p.thresholds != nullptr) {
    ApplyThresholds(output_, p);
    const auto buf = std::make_unique<QUANTIZED_PACKED[]>(out_size * p.n_bit / CHAR_BIT);
    pack_16bit(p.device_output_buf, buf.get(), out_size);
    const std::size_t b = 32;
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>::tensor_info_t<std::size_t> buf_shape = {
      oh, ow, (oc + b - 1) / b, p.n_bit, b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl> buf_tensor(buf.get(), buf_shape);
    TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>::tensor_info_t<std::size_t> out_shape = {
      (oc + b - 1) / b,
      oh,
      ow,
      p.n_bit,
      b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl> out((QUANTIZED_PACKED*)p.device_output_buf, out_shape);
    convert_tensor(buf_tensor, out);
  } else {
    const std::size_t b = 32;
    const auto buf = std::make_unique<BIN_CONV_OUTPUT[]>(out_size);
    std::copy(p.device_output_buf, p.device_output_buf + out_size, buf.get());
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>::tensor_info_t<std::size_t> buf_shape = {
      oh, ow, oc 
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC> buf_tensor(buf.get(), buf_shape);
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl>::tensor_info_t<std::size_t> out_shape = {
      (oc + b - 1) / b, oh, ow, b
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl> out(p.device_output_buf, out_shape);
    convert_tensor(buf_tensor, out);
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

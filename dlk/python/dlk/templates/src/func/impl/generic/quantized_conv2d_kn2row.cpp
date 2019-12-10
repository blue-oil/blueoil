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

static const auto kn2row_buf = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_SIZE_KN2ROW_BUFFER_PER_LAYER);
static const auto apply_threshold_buf = std::make_unique<QUANTIZED_PACKED[]>(MAX_SIZE_QOUTPUTS_PER_LAYER * sizeof(QUANTIZED_PACKED));
static const auto convert_buf = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_SIZE_OUTPUTS_PER_LAYER);

void QuantizedConv2DKn2Row(const kn2row_input_t& input,
                                  const kernel_t& kernel,
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

  assert(ih * iw == oh * ow);

  Measurement::Start("quantized-kn2row");

  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, oc, ih * iw);
  auto kernel_ = MatrixView<QUANTIZED_PACKED_KERNEL, MatrixOrder::RowMajor>(
      kernel.data(), oc * kh * kw, ic / 32);
  if (kh == kw && kw == 3) {
    std::fill(p.device_output_buf, p.device_output_buf + oc * oh * ow, 0);
    for (std::size_t offset = 0; offset < ih * iw; offset += MAX_SIZE_KN2ROW_COL_BLOCK) {
      const auto col_block = std::min(static_cast<std::size_t>(MAX_SIZE_KN2ROW_COL_BLOCK), ih * iw - offset);
      auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
          input.data() + offset * ic / 16, ic / 16, col_block);
      auto buf_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
          kn2row_buf.get(), oc * kh * kw, col_block);

      quantized_matrix_multiplication(kernel_, input_, buf_);
      matrix_shift_add(buf_, output_, p.normal_conv_params, offset);
    }
  } else if (kh == kw && kw == 1) {
    auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
        input.data(), ic / 16, ih * iw);
    auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
        p.device_output_buf, oc, ih * iw);
    quantized_matrix_multiplication(kernel_, input_, output_);
  } else {
    std::cerr << "Only 1x1 or 3x3 convolutions are supported." << std::endl;
    assert(false);
  }

  const auto out_size = oc * oh * ow;
  if (p.thresholds != nullptr) {
    ApplyThresholds(output_, p);
    pack_16bit(p.device_output_buf, apply_threshold_buf.get(), out_size);
    const std::size_t b = 32;
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>::tensor_info_t<std::size_t> buf_shape = {
      oh, ow, (oc + b - 1) / b, p.n_bit, b
    };
    TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl> buf_tensor(apply_threshold_buf.get(), buf_shape);
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
    std::copy(p.device_output_buf, p.device_output_buf + out_size, convert_buf.get());
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>::tensor_info_t<std::size_t> buf_shape = {
      oh, ow, oc 
    };
    TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC> buf_tensor(convert_buf.get(), buf_shape);
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

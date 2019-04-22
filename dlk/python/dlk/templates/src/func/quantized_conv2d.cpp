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
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>

#include "global.h"
#include "func/impl/apply_thresholds.h"
#include "func/impl/quantized_conv2d_dim2col.h"
#include "func/impl/quantized_conv2d_tiling.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "func/quantized_conv2d.h"
#include "time_measurement.h"

namespace {

// temporary:

void func_linear_to_float(BIN_CONV_OUTPUT input[], T_INT nbit,
                          T_FLOAT max_value, T_FLOAT output[], T_UINT in_height,
                          T_UINT in_width, T_UINT in_depth,
                          T_UINT in_channel = 1) {
  T_FLOAT n = (1 << nbit) - 1;
  unsigned num_elems = in_height * in_width * in_depth * in_channel;

  for (unsigned i = 0; i < num_elems; i++) {
    T_FLOAT tmp = (T_FLOAT)input[i];
    tmp = tmp / n;
    output[i] = tmp * max_value;
  }
}

void QuantizedConv2D(QUANTIZED_NOT_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                            binary_convolution_parameters p) {
  constexpr T_UINT TilingInTypeBitWidth = CHAR_BIT * sizeof(QUANTIZED_PACKED);
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
      dlk::impl::QuantizedConv2DTiling(input, kernel, p);
#else
      dlk::impl::QuantizedConv2DKn2Row(input, kernel, p);
#endif
    } else {
      dlk::impl::QuantizedConv2DKn2Row(input, kernel, p);
    }
  } else {
    dlk::impl::QuantizedConv2DIm2Col(input, kernel, p);
  }
}

} // namespace

void func_QuantizedConv2D(QUANTIZED_NOT_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                          T_FLOAT output[], T_FLOAT scaling_factor,
                          binary_convolution_parameters p) {
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
    output[i] = (scaling_factor * post_qtz_factor) * p.device_output_buf[i];
  }

  Measurement::Stop();
}

void func_QuantizedConv2D(QUANTIZED_NOT_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                          T_FLOAT output[], T_FLOAT scaling_factor[],
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
      output[idx] =
          (scaling_factor[c] * post_qtz_factor) * static_cast<T_FLOAT>(out);
    }
  }

  Measurement::Stop();
}

void func_QuantizedConv2DWithThreshold(QUANTIZED_NOT_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[],
                                       QUANTIZED_NOT_PACKED output[],
                                       T_FLOAT scaling_factor,
                                       binary_convolution_parameters p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  for (unsigned i = 0; i < out_elems; ++i) {
    output[i] = p.device_output_buf[i];
  }
}

void func_QuantizedConv2DWithThreshold(QUANTIZED_NOT_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[], T_FLOAT output[],
                                       T_FLOAT scaling_factor,
                                       binary_convolution_parameters p) {
  QuantizedConv2D(input, kernel, p);

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  func_linear_to_float(p.device_output_buf, p.n_bit, p.max_value, output,
                       p.normal_conv_params.output_height,
                       p.normal_conv_params.output_width,
                       p.normal_conv_params.output_channels);
}

void func_QuantizedConv2DWithThreshold(QUANTIZED_NOT_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[],
                                       QUANTIZED_NOT_PACKED output[],
                                       T_FLOAT scaling_factor[],
                                       binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

void func_QuantizedConv2DWithThreshold(QUANTIZED_NOT_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[], T_FLOAT output[],
                                       T_FLOAT scaling_factor[],
                                       binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

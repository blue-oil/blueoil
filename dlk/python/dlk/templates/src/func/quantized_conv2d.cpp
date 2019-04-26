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
#include "pack_input_to_qwords.h"

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

void QuantizedConv2D(QUANTIZED_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
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
  // else
  //   std::memset((void *)p.device_output_buf, 0, size * sizeof(BIN_CONV_OUTPUT));

  if ((kh == 3 && kw == 3 && padding == 1) ||
      (kh == 1 && kw == 1 && padding == 0)) {
    if ((ic % TilingInTypeBitWidth) == 0) {
#if defined(USE_NEON) && !defined(RUN_ON_FPGA)
      dlk::impl::QuantizedConv2DTiling(input, kernel, p);
#else
      dlk::impl::TCAConv2d(input, kernel, p);
#endif
    } else {
      dlk::impl::TCAConv2d(input, kernel, p);
    }
  } else {
    ;//dlk::impl::QuantizedConv2DIm2Col(input, kernel, p);
  }
}

} // namespace

void func_QuantizedConv2D(QUANTIZED_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                          T_FLOAT output[], T_FLOAT scaling_factor,
                          binary_convolution_parameters p) {

  unsigned in_elems_wh = p.normal_conv_params.input_height * p.normal_conv_params.input_width;
  unsigned in_elems = (p.normal_conv_params.kernel_depth > 32 ? (in_elems_wh * p.normal_conv_params.kernel_depth) / 16 : in_elems_wh * 2);
  static T_UINT in_counter = 0;
  // write_to_file("out/qconv_input_quantized_packed_ol1_", in_counter++, input, in_elems);

  Measurement::Start("func_QuantizedConv2D");
  Measurement::Start("QuantizedConv2D");

  QuantizedConv2D(input, kernel, p);

  Measurement::Stop();

  Measurement::Start("QuantizedConv2D_ApplyScalingFactor");

  unsigned out_elems = p.normal_conv_params.output_height *
                       p.normal_conv_params.output_width *
                       p.normal_conv_params.output_channels;

  static T_UINT out_counter = 0;
  // write_to_file("out/qconv_output_16bit_ol1_", out_counter++, p.device_output_buf, out_elems);


  // temporary: (2^n - 1) * (max - min)
  const T_FLOAT post_qtz_factor = 2.0f / 3.0f;
  int b = 32;
  auto& ncp(p.normal_conv_params);

  if(ncp.output_channels > b) {
      int out_index = 0;
      for (int h = 0; h < ncp.output_height; h++)
      for (int w = 0; w < ncp.output_width; w++)
      for (int s = 0; s < ncp.output_channels / b; s++)
      for (int d = 0; d < b; d++)
        output[out_index++] = (scaling_factor * post_qtz_factor) * p.device_output_buf[h * (ncp.output_channels * ncp.input_width) + w * ncp.output_channels + s * (ncp.input_height * ncp.input_width * b) + d];
  }
  else {
      int tca_channels = ((ncp.output_channels + b - 1) / b) * b;
      int out_index = 0;
      for (int h = 0; h < ncp.output_height; h++)
      for (int w = 0; w < ncp.output_width; w++)
      for (int d = 0; d < ncp.output_channels; d++)
        output[out_index++] = (scaling_factor * post_qtz_factor) * p.device_output_buf[h * (tca_channels * ncp.input_width) + w * tca_channels + d];
  }

  Measurement::Stop();
  Measurement::Stop();
}

void func_QuantizedConv2D(QUANTIZED_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                          T_FLOAT output[], T_FLOAT scaling_factor[],
                          binary_convolution_parameters p) {
  Measurement::Start("func_QuantizedConv2D");
  Measurement::Start("QuantizedConv2D");

  unsigned in_elems_wh = p.normal_conv_params.input_height * p.normal_conv_params.input_width;
  unsigned in_elems = (p.normal_conv_params.kernel_depth > 32 ? (in_elems_wh * p.normal_conv_params.kernel_depth) / 16 : in_elems_wh * 2);
  static T_UINT in_counter = 0;
  // write_to_file("out/qconv_input_quantized_packed_ol2_", in_counter++, input, in_elems);


  QuantizedConv2D(input, kernel, p);

  Measurement::Stop();


  unsigned out_elems = p.normal_conv_params.output_height * p.normal_conv_params.output_width;
  unsigned out_channels = p.normal_conv_params.output_channels;

  int b = 32;
  auto& ncp(p.normal_conv_params);
  int tca_channels = ((ncp.output_channels + b - 1) / b) * b;

  static T_UINT out_counter = 0;
  // write_to_file("out/qconv_output_16bit_ol2_", out_counter++, p.device_output_buf, out_elems * tca_channels);

  // temporary: (2^n - 1) * (max - min)
  T_FLOAT post_qtz_factor = 2.0 / 3.0;

  if (ncp.output_channels > b) {
      Measurement::Start("QuantizedConv2D_ChangeOutputLayout");
      // XXX: assumes that ncp.output_channels % b == 0
      int out_index = 0;
      for (int h = 0; h < ncp.output_height; h++)
      for (int w = 0; w < ncp.output_width; w++)
      for (int s = 0; s < ncp.output_channels / b; s++)
      for (int d = 0; d < b; d++) {
        auto buf_index = h * (b * ncp.output_width) + w * b + s * (ncp.output_height * ncp.output_width * b) + d;
        output[out_index++] = static_cast<float>(p.device_output_buf[buf_index]);
      }
      Measurement::Stop();

      // write_to_file((std::string("out/") + p.debug_name).c_str(), 0, output, out_elems * p.normal_conv_params.output_channels);

      Measurement::Start("QuantizedConv2D_ApplyScalingFactor");
      for (unsigned i = 0; i < out_elems; ++i) {
        for (unsigned c = 0; c < out_channels; c++) {
          unsigned idx = i * out_channels + c;
          output[idx] = (scaling_factor[c] * post_qtz_factor) * output[idx];
        }
      }
      Measurement::Stop();
  }
  else {
      Measurement::Start("QuantizedConv2D_RemoveChannels");
      int tmp_index = 0;
      auto* tmp_output = new T_FLOAT[out_elems * p.normal_conv_params.output_channels];
      for (int h = 0; h < ncp.output_height; h++)
      for (int w = 0; w < ncp.output_width; w++)
      for (int d = 0; d < ncp.output_channels; d++) {
        tmp_output[tmp_index++] = p.device_output_buf[h * (tca_channels * ncp.input_width) + w * tca_channels + d];
      }

      // write_to_file((std::string("out/") + p.debug_name).c_str(), 0, tmp_output, out_elems * p.normal_conv_params.output_channels);
      delete [] tmp_output;
      Measurement::Stop();

      Measurement::Start("QuantizedConv2D_ApplyScalingFactor");
      int out_index = 0;
      for (int h = 0; h < ncp.output_height; h++)
      for (int w = 0; w < ncp.output_width; w++)
      for (int d = 0; d < ncp.output_channels; d++)
        output[out_index++] = (scaling_factor[d] * post_qtz_factor) * p.device_output_buf[h * (tca_channels * ncp.input_width) + w * tca_channels + d];
      Measurement::Stop();
  }

  //// write_to_file("out/qconv_output_16bit_ol2_final_", out_counter_final++, output, out_elems * p.normal_conv_params.output_channels);
  Measurement::Stop();


}

void func_QuantizedConv2DWithThreshold(QUANTIZED_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[],
                                       QUANTIZED_PACKED output[],
                                       T_FLOAT scaling_factor,
                                       binary_convolution_parameters p) {

  auto& ncp(p.normal_conv_params);

  unsigned in_elems = ncp.input_height * ncp.input_width * ncp.kernel_depth;
  unsigned out_elems = ncp.output_height * ncp.output_width * ncp.output_channels;

  // int b = 32;
  // int packed_input_depth = (ncp.kernel_depth / 32) * 2;
  // int packed_b = (b / 32) * 2;

  const T_UINT b = 32;
  const T_UINT out_c = ((ncp.output_channels + b - 1) / b) * b;
  int packed_output_depth = (out_c / 32) * 2;
  int packed_b = (b / 32) * 2;


  // TODO: replace input with 'input_tca_layout' when ready
  QuantizedConv2D(input, kernel, p);
  if (ncp.output_channels > b) {

      Measurement::Start("QuantizedConv2D_ChangeOutputLayout");
      // XXX: assumes that ncp.output_channels % b == 0
      int out_index = 0;
      for (int h = 0; h < ncp.output_height; h++)
      for (int w = 0; w < ncp.output_width; w++)
      for (int s = 0; s < out_c / b; s++)
      for (int d = 0; d < packed_b; d++) {
        auto buf_index = h * (packed_b * ncp.output_width) + w * packed_b + s * (ncp.output_height * ncp.output_width * packed_b) + d;
        output[out_index++] = QUANTIZED_PACKED(reinterpret_cast<volatile int32_t*>(p.device_output_buf)[buf_index]);
      }
      Measurement::Stop();

      // write_to_file((std::string("out/") + p.debug_name).c_str(), 0, output, out_elems * p.normal_conv_params.output_channels);
  }
  else {
      Measurement::Start("Copy");
      int tmp_index = 0;
      int number_of_elements = ncp.output_height * ncp.output_width * packed_output_depth;

      for (int i = 0; i < number_of_elements; i++) {
        output[i] = QUANTIZED_PACKED(reinterpret_cast<volatile int32_t*>(p.device_output_buf)[i]);
      }


      // write_to_file((std::string("out/") + p.debug_name).c_str(), 0, tmp_output, out_elems * p.normal_conv_params.output_channels);
      Measurement::Stop();
  }
}

void func_QuantizedConv2DWithThreshold(QUANTIZED_PACKED input[],
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

void func_QuantizedConv2DWithThreshold(QUANTIZED_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[],
                                       QUANTIZED_PACKED output[],
                                       T_FLOAT scaling_factor[],
                                       binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}

void func_QuantizedConv2DWithThreshold(QUANTIZED_PACKED input[],
                                       QUANTIZED_PACKED_KERNEL kernel[], T_FLOAT output[],
                                       T_FLOAT scaling_factor[],
                                       binary_convolution_parameters p) {
  func_QuantizedConv2DWithThreshold(input, kernel, output, scaling_factor[0],
                                    p);
}


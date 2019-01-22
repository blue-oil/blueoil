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

#include "global.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "matrix_view.h"
#include "matrix/quantized_multiplication.h"
#include "matrix/shift_add.h"
#include "pack_input_to_qwords.h"
#include "time_measurement.h"

namespace {

// kernel format converter
// ohwi : oc kh kw ic, hwoi: kh kw oc ic
void quantized_ohwi_to_hwoi(const T_UINT ohwi[], T_UINT hwoi[], const struct binary_convolution_parameters& p) {
   Measurement::Start("quantized_ohwi_to_hwoi");

   int ic = p.normal_conv_params.kernel_depth / 32;
   int oc = p.normal_conv_params.output_channels;
   int kh = p.normal_conv_params.kernel_height;
   int kw = p.normal_conv_params.kernel_width;

   for (unsigned int i = 0; i < kh*kw; ++i) {
     for (unsigned int j = 0; j < oc; ++j) {
       for (unsigned int k = 0; k < ic; ++k) {
         hwoi[i*oc*ic + j*ic + k] = ohwi[i*ic + j*ic*kh*kw + k];
       }
     }
   }

   Measurement::Stop();
 }

void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");

  for (unsigned int i = 0; i < result.rows(); ++i) {
    for (unsigned int j = 0; j < result.cols(); ++j) {
      BIN_CONV_OUTPUT d = *result.data(i, j);
      T_INT ts0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i];
      T_INT ts1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
      T_INT ts2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
      T_INT flag = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
      BIN_CONV_OUTPUT new_d;

      if (flag == 1) { // increasing function
        if (d < ts0)
          new_d = 0;
        else if (d < ts1)
          new_d = 1;
        else if (d < ts2)
          new_d = 2;
        else
          new_d = 3;
      } else if (flag == -1) { // decreasing function
        if (d > ts2)
          new_d = 0;
        else if (d > ts1)
          new_d = 1;
        else if (d > ts0)
          new_d = 2;
        else
          new_d = 3;
      } else {                            // constant function
        new_d = flag - 2;                 // note: 2 is a magic number!
        assert(0 <= new_d && new_d <= 3); // unsinged 2bits
      }
      *result.data(i, j) = new_d;
    }
  }

  Measurement::Stop();
}

} // namespace

namespace dlk {

namespace impl {

void QuantizedConv2DKn2Row_3x3(QUANTIZED_NOT_PACKED input[],
			       const T_UINT kernel[],
			       const binary_convolution_parameters &p) {
  using namespace dlk;

  convolution_parameters cp = p.normal_conv_params;
  const T_UINT out_c = cp.output_channels;

  int ic = p.normal_conv_params.kernel_depth;
  int ih = p.normal_conv_params.input_height;
  int iw = p.normal_conv_params.input_width;
  int oc = p.normal_conv_params.output_channels;
  int oh = p.normal_conv_params.output_height;
  int ow = p.normal_conv_params.output_width;
  const int kh = dlk::impl::KN2ROW_FACTOR_3x3;
  const int kw = dlk::impl::KN2ROW_FACTOR_3x3;

  assert(ih * iw == oh * ow);
  assert(MAX_SIZE_IM2COL_INPUTS_PER_LAYER >= ic * kh * kw * ih * iw);

  Measurement::Start("quantized-kn2row");

  int kernel_buf_size = kh * kw * ic * oc / 32;
  auto kernel_hwoi = new T_UINT[kernel_buf_size]();
  quantized_ohwi_to_hwoi(kernel, kernel_hwoi, p);

  pack_input_to_qwords(input, p.device_input_buf, ih * iw * ic, 2);
  auto kernel_ = MatrixView<T_UINT, MatrixOrder::RowMajor>(
      kernel_hwoi, oc * kh * kw, ic / 32);
  auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
      p.device_input_buf, ic / 16, ih * iw);
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, oc, ih * iw);

  unsigned bufsize = oc * kh * kw * ih * iw;
  BIN_CONV_OUTPUT *kn2row_buf = new BIN_CONV_OUTPUT[bufsize]();
  std::memset((void*)kn2row_buf, 0, bufsize);
  auto buf_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
	  kn2row_buf, oc * kh * kw, ih * iw);
  
  quantized_matrix_multiplication(kernel_, input_, buf_);
  matrix_shift_add(buf_, output_, p.normal_conv_params);
  delete[] kn2row_buf;
  delete[] kernel_hwoi;

  if (p.thresholds != NULL) {
    ApplyThresholds(output_, p);
  }

  Measurement::Stop();
}

void QuantizedConv2DKn2Row_1x1(QUANTIZED_NOT_PACKED input[],
			       const T_UINT kernel[],
			       const binary_convolution_parameters &p) {
  using namespace dlk;

  convolution_parameters cp = p.normal_conv_params;
  const T_UINT out_c = cp.output_channels;

  int ic = p.normal_conv_params.kernel_depth;
  int ih = p.normal_conv_params.input_height;
  int iw = p.normal_conv_params.input_width;
  int oc = p.normal_conv_params.output_channels;
  int oh = p.normal_conv_params.output_height;
  int ow = p.normal_conv_params.output_width;
  const int kh = dlk::impl::KN2ROW_FACTOR_1x1;
  const int kw = dlk::impl::KN2ROW_FACTOR_1x1;

  assert(ih * iw == oh * ow);
  assert(MAX_SIZE_IM2COL_INPUTS_PER_LAYER >= ic * kh * kw * ih * iw);

  Measurement::Start("quantized-kn2row");

  int kernel_buf_size = kh * kw * ic * oc / 32;
  auto kernel_hwoi = new T_UINT[kernel_buf_size]();
  quantized_ohwi_to_hwoi(kernel, kernel_hwoi, p);

  pack_input_to_qwords(input, p.device_input_buf, ih * iw * ic, 2);
  auto kernel_ = MatrixView<T_UINT, MatrixOrder::RowMajor>(
      kernel_hwoi, oc * kh * kw, ic / 32);
  auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
      p.device_input_buf, ic / 16, ih * iw);
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, oc, ih * iw);

  quantized_matrix_multiplication(kernel_, input_, output_);
  delete[] kernel_hwoi;

  if (p.thresholds != NULL) {
    ApplyThresholds(output_, p);
  }

  Measurement::Stop();
}  

} // namespace impl

} // namespace dlk

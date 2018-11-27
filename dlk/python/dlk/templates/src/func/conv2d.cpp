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
#include "func/conv2d.h"
#include "matrix_view.h"
#include "matrix/shift_add.h"
#include "matrix/multiplication.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

namespace {

template<typename T, typename U>
void conv3x3_kn2row(T input[],
                    T kernels[],
                    U output[],
                    struct convolution_parameters& p) {
  const int ic = p.kernel_depth;
  const int oc = p.output_channels;
  const int kh = p.kernel_height;
  const int kw = p.kernel_width;
  const int ih = p.input_height;
  const int iw = p.input_width;
  const int oh = p.output_height;
  const int ow = p.output_width;

  // assertions
  assert(ih * iw == oh * ow);
  assert(kh == 3 && kw == 3);
  assert(MAX_SIZE_KN2ROW_BUFFER_PER_LAYER >= oc * kh * kw * ih * iw);

  Measurement::Start("kn2row");
  Measurement::Start("kn2row-buf");

  assert(p.input_height > 0);
  assert(p.input_width > 0);

  static U buf[MAX_SIZE_KN2ROW_BUFFER_PER_LAYER];

  Measurement::Stop();

  auto kernels_ = dlk::MatrixView<T, dlk::MatrixOrder::RowMajor>(kernels, oc * kh * kw, ic);
  auto input_ = dlk::MatrixView<T, dlk::MatrixOrder::ColMajor>(input, ic, p.input_height * p.input_width);
  auto buf_ = dlk::MatrixView<U, dlk::MatrixOrder::ColMajor>(buf, oc * kh * kw, p.input_height * p.input_width);
  auto output_ = dlk::MatrixView<U, dlk::MatrixOrder::ColMajor>(output, oc, p.input_height * p.input_width);

  dlk::matrix_multiplication(kernels_, input_, buf_);
  dlk::matrix_shift_add(buf_, output_, p);

  Measurement::Stop();
}

// kernel format converter
// ohwi : oc kh kw ic, hwoi: kh kw oc ic
template<typename T>
void ohwi_to_hwoi(const T ohwi[], T hwoi[], const struct convolution_parameters& p) {
  int ic = p.kernel_depth;
  int oc = p.output_channels;
  int kh = p.kernel_height;
  int kw = p.kernel_width;

  for (unsigned int i = 0; i < kh*kw; ++i) {
    for (unsigned int j = 0; j < oc; ++j) {
      for (unsigned int k = 0; k < ic; ++k) {
        hwoi[i*oc*ic + j*ic + k] = ohwi[i*ic + j*ic*kh*kw + k];
      }
    }
  }
 }

template<typename T, typename U>
void conv1x1_kn2row(T input[],
                    T kernels[],
                    U output[],
                    struct convolution_parameters& p) {
  int ic = p.kernel_depth;
  int oc = p.output_channels;
  int kh = p.kernel_height;
  int kw = p.kernel_width;

  Measurement::Start("kn2row-1x1");


   assert(p.input_height > 0);
   assert(p.input_width > 0);
  
   auto kernels_ = dlk::MatrixView<T, dlk::MatrixOrder::RowMajor>(kernels, oc * kh * kw, ic);
   auto input_ = dlk::MatrixView<T, dlk::MatrixOrder::ColMajor>(input, ic, p.input_height * p.input_width);
   auto output_ = dlk::MatrixView<U, dlk::MatrixOrder::ColMajor>(output, oc, p.input_height * p.input_width);

   dlk::matrix_multiplication(kernels_, input_, output_);

   Measurement::Stop();
}

template<typename T>
void conv_general(
  T input[],
  T kernels[],
  T output[],
  struct convolution_parameters p)
{
  for(T_UINT wi = 0; wi < p.output_height; wi++)
  for(T_UINT wj = 0; wj < p.output_width; wj++)
  {
    for(T_UINT kernel_id = 0; kernel_id < p.output_channels; kernel_id++)
    {
      T_UINT kernel_offset = kernel_id * p.kernel_elements;

      T out = 0;
      T_UINT current_kernel_index = 0;

      bool inside_row, inside_col;

      for(T_UINT ki = 0; ki < p.kernel_height; ki++)
      {
        T_INT row = (wi * p.stride_along_height) - p.padding + ki;
	if (row < 0 || row >= p.input_height) { current_kernel_index += p.kernel_width * p.kernel_depth; continue; }
				   
        for(T_UINT kj = 0; kj < p.kernel_width; kj++)
        {
          T_INT col = (wj * p.stride_along_width)  - p.padding + kj;
	  if (col < 0 || col >= p.input_width) { current_kernel_index += p.kernel_depth; continue; }

          for(T_UINT kz = 0; kz < p.kernel_depth; kz++)
          {
	    unsigned in_idx = row * (p.input_width * p.kernel_depth) + col * (p.kernel_depth) + kz;
	    unsigned k_idx = current_kernel_index + kernel_offset;	    

	    T in_data = input[in_idx];
	    T k_data = kernels[k_idx];
	    
	    out += in_data * k_data;
            current_kernel_index++;
          }
        }
      }

      unsigned out_idx = wi * (p.output_channels * p.output_width) + wj * (p.output_channels) + kernel_id;
      output[out_idx] = out;
    }
  }
}

template<typename T>
void convolution(
  T input[],
  T kernels[],
  T output[],
  struct convolution_parameters p)
{
  // use special implementation for 1x1 conv
  if (p.kernel_height == 1 && p.kernel_width == 1 && p.padding == 0) {
    conv1x1_kn2row(input, kernels, output, p);
    return;
  } else if (p.kernel_height == 3 && p.kernel_width == 3 && p.padding == 1) {
    int kernels_size = p.kernel_height * p.kernel_width * p.kernel_depth * p.output_channels;
    T* kernels_hwoi = new T[kernels_size];
    ohwi_to_hwoi(kernels, kernels_hwoi, p);
    conv3x3_kn2row(input, kernels_hwoi, output, p);      
    delete[] kernels_hwoi;
    return;
  }

  // otherwise, use hand-written implementation
  conv_general(input, kernels, output, p);
}

} // namespace

void func_Conv2D(T_FLOAT input[], T_FLOAT weights[], T_FLOAT output[],
                 struct convolution_parameters p, T_UINT out_height,
                 T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("Convolution");

  unsigned k_elems = p.kernel_height * p.kernel_width * p.kernel_depth;
  unsigned in_elems = out_height * out_width * k_elems;

  convolution(input, weights, output, p);

  Measurement::Stop();
}

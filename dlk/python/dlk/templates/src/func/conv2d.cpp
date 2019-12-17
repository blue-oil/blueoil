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

#include <memory>

#include "global.h"
#include "func/conv2d.h"
#include "matrix_view.h"
#include "matrix/shift_add.h"
#include "matrix/multiplication.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

namespace {

template<typename T, typename U>
void conv_nxn_kn2row(const TensorView<T, MemoryLayout::NHWC>& input,
    const TensorView<T, MemoryLayout::HWOI>& kernels,
    const TensorView<U, MemoryLayout::NHWC>& output,
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
  assert(kh == kw);
  assert(3 <= kh && kh <= 5);

  // need to initialize output
  std::memset(output.data(), 0, oc * ih * iw * sizeof(U));

  Measurement::Start("kn2row");
  Measurement::Start("kn2row-buf");

  assert(p.input_height > 0);
  assert(p.input_width > 0);

  T* buf = reinterpret_cast<T*>(p.temporary_buf) + MAX_SIZE_KERNELS_PER_LAYER; // offset comes from kernel layout convert buffer
  BYTE* matmul_buf = reinterpret_cast<BYTE*>(buf + MAX_SIZE_KN2ROW_BUFFER_PER_LAYER);

  Measurement::Stop();

  auto kernels_ = dlk::MatrixView<T, dlk::MatrixOrder::RowMajor>(kernels.data(), oc * kh * kw, ic);
  auto output_ = dlk::MatrixView<U, dlk::MatrixOrder::ColMajor>(output.data(), oc, p.input_height * p.input_width);
  for (std::size_t offset = 0; offset < ih * iw; offset += MAX_SIZE_KN2ROW_COL_BLOCK) {
    auto col_block = std::min(static_cast<std::size_t>(MAX_SIZE_KN2ROW_COL_BLOCK), ih * iw - offset);
    auto input_ = dlk::MatrixView<T, dlk::MatrixOrder::ColMajor>(input.data() + ic * offset, ic, col_block);
    auto buf_ = dlk::MatrixView<U, dlk::MatrixOrder::ColMajor>(buf, oc * kh * kw, col_block);

    dlk::matrix_multiplication(kernels_, input_, buf_, matmul_buf);
    dlk::matrix_shift_add(buf_, output_, p, offset);
  }

  Measurement::Stop();
}

// kernel format converter
// ohwi : oc kh kw ic, hwoi: kh kw oc ic
template<typename T>
void ohwi_to_hwoi(const TensorView<T, MemoryLayout::OHWI>& ohwi,
    const TensorView<T, MemoryLayout::HWOI>& hwoi,
    const struct convolution_parameters& p) {
  int ic = p.kernel_depth;
  int oc = p.output_channels;
  int kh = p.kernel_height;
  int kw = p.kernel_width;

  for (unsigned int r = 0; r < kh; ++r) {
    for (unsigned int c = 0; c < kw; ++c) {
      for (unsigned int j = 0; j < oc; ++j) {
        for (unsigned int k = 0; k < ic; ++k) {
          hwoi(r, c, j, k) = ohwi(j, r, c, k);
        }
      }
    }
  }
 }

template<typename T, typename U>
void conv1x1_kn2row(const TensorView<T, MemoryLayout::NHWC>& input,
                    const TensorView<T, MemoryLayout::OHWI>& kernels,
                    const TensorView<U, MemoryLayout::NHWC>& output,
                    struct convolution_parameters& p) {
  int ic = p.kernel_depth;
  int oc = p.output_channels;
  int kh = p.kernel_height;
  int kw = p.kernel_width;

  Measurement::Start("kn2row-1x1");


  assert(p.input_height > 0);
  assert(p.input_width > 0);

  auto kernels_ = dlk::MatrixView<T, dlk::MatrixOrder::RowMajor>(kernels.data(), oc * kh * kw, ic);
  auto input_ = dlk::MatrixView<T, dlk::MatrixOrder::ColMajor>(input.data(), ic, p.input_height * p.input_width);
  auto output_ = dlk::MatrixView<U, dlk::MatrixOrder::ColMajor>(output.data(), oc, p.input_height * p.input_width);

  // offset comes from kernel layout convert buffer
  BYTE* matmul_buf = p.temporary_buf + MAX_SIZE_KERNELS_PER_LAYER * sizeof(T);
  dlk::matrix_multiplication(kernels_, input_, output_, matmul_buf);

  Measurement::Stop();
}

template<typename T>
void conv_general(
  const TensorView<T, MemoryLayout::NHWC>& input,
  const TensorView<T, MemoryLayout::OHWI>& kernels,
  const TensorView<T, MemoryLayout::NHWC>& output,
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
        inside_row = (row >= 0 && row < (T_INT) p.input_height);

        for(T_UINT kj = 0; kj < p.kernel_width; kj++)
        {
          T_INT col = (wj * p.stride_along_width)  - p.padding + kj;
          inside_col = (col >= 0 && col < (T_INT) p.input_width);

          for(T_UINT kz = 0; kz < p.kernel_depth; kz++)
          {
            if (inside_row && inside_col) {
              unsigned k_idx = current_kernel_index + kernel_offset;

              T in_data = input(0, row, col, kz);
              T k_data = kernels(kernel_id, ki, kj, kz);

              out += in_data * k_data;
            }
            current_kernel_index++;
          }
        }
      }

      output(0, wi, wj, kernel_id) = out;
    }
  }
}

template<typename T>
void convolution(
  const TensorView<T, MemoryLayout::NHWC>& input,
  const TensorView<T, MemoryLayout::OHWI>& kernels,
  const TensorView<T, MemoryLayout::NHWC>& output,
  struct convolution_parameters p)
{
  // use special implementation for 1x1 conv
  if (p.kernel_height == 1 && p.kernel_width == 1 && p.padding == 0) {
    int kernels_size = p.kernel_height * p.kernel_width * p.kernel_depth * p.output_channels;
    conv1x1_kn2row(input, kernels, output, p);
    return;
  } else if (p.kernel_height == p.kernel_width && 3 <= p.kernel_height && p.kernel_height <= 5 && p.padding == p.kernel_height / 2) {
    int kernels_size = p.kernel_height * p.kernel_width * p.kernel_depth * p.output_channels;
    T* buf = reinterpret_cast<T*>(p.temporary_buf);
    using hwoi_t = TensorView<T, MemoryLayout::HWOI>;
    typename hwoi_t::template tensor_info_t<std::size_t> hwoi_shape = {
      p.kernel_height,
      p.kernel_width,
      p.output_channels,
      p.kernel_depth
    };
    hwoi_t kernels_hwoi(buf, hwoi_shape);
    ohwi_to_hwoi(kernels, kernels_hwoi, p);
    conv_nxn_kn2row(input, kernels_hwoi, output, p);
    return;
  }

  // otherwise, use hand-written implementation
  conv_general(input, kernels, output, p);
}

} // namespace

void func_Conv2D(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::OHWI>& weights,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    struct convolution_parameters p) {
  Measurement::Start("Convolution");

  unsigned out_height = output.get_shape()[1];
  unsigned out_width = output.get_shape()[2];
  unsigned out_depth = output.get_shape()[3];
  unsigned k_elems = p.kernel_height * p.kernel_width * p.kernel_depth;
  unsigned in_elems = out_height * out_width * k_elems;

  convolution(input, weights, output, p);

  Measurement::Stop();
}

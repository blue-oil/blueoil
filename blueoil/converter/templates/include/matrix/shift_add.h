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

#ifndef DLK_MATRIX_SHIFT_ADD_H_INCLUDED
#define DLK_MATRIX_SHIFT_ADD_H_INCLUDED

#include "matrix_view.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

namespace dlk {

template<typename T>
void matrix_shift_add(MatrixView<T, MatrixOrder::ColMajor>& buf,
                      MatrixView<T, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p,
                      const int block_offset) {
  Measurement::Start("matrix_shift_add");

  const std::ptrdiff_t h = p.input_height;
  const std::ptrdiff_t w = p.input_width;
  const std::ptrdiff_t oc = p.output_channels;
  const std::ptrdiff_t kh = p.kernel_height;
  const std::ptrdiff_t kw = p.kernel_width;
  const std::ptrdiff_t col_block = buf.cols();
  const std::ptrdiff_t pad = p.padding;

  // only 3x3 or 5x5 kernel is supported.
  assert(kh == kw);
  assert(kh % 2 == 1);
  assert(3 <= kh && kh <= 5);

  for (int k = 0; k < col_block; ++k) {
    const auto true_k = k + block_offset;
    const auto row = true_k / w;
    const auto col = true_k % w;
    for (unsigned int i = 0; i < kh * kw; ++i) {
      int kr = i / kw;
      int kc = i % kw;
      if (row - kr + pad < 0 || row - kr + pad >= h || col - kc + pad < 0 || col - kc + pad >= w) continue;

      int offset = (kr - pad) * w + (kc - pad);
      T* r = result.data(0, true_k - offset);
      T* b = buf.data(i*oc, k);

      for (unsigned int j = 0; j < oc; ++j) {
        r[j] += b[j];
      }
    }
  }

  Measurement::Stop();
}

template<>
void matrix_shift_add(MatrixView<int32_t, MatrixOrder::ColMajor>& buf,
                      MatrixView<int32_t, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p,
                      const int block_offset);
template<>
void matrix_shift_add(MatrixView<float, MatrixOrder::ColMajor>& buf,
                      MatrixView<float, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p,
                      const int block_offset);

} // namespace dlk

#endif // DLK_MATRIX_SHIFT_ADD_H_INCLUDED

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

inline bool is_first_column(int j, int w) {
  return (j % w == 0);
}

inline bool is_last_column(int j, int w) {
  return (j % w == (w - 1));
}

 // 3x3 matrix
 /* A B C */
 /* D E F */
 /* G H I */

// is the right most column for the kernel matrix?
inline bool is_cfi(int i, int oc) {
  return int(i / oc) == 2 or int(i / oc) == 5 or int(i / oc) == 8;
}

// is the left most column for the kernel matrix?
inline bool is_adg(int i, int oc) {
  return int(i / oc) == 0 or int(i / oc) == 3 or int(i / oc) == 6;
}

// Note: this function is only for 3x3 kernel
inline int calc_offset(int i, int w) {
  switch (i) {
  case 0:
    return w+1;
  case 1:
    return w;
  case 2:
    return w-1;
  case 3:
    return 1;
  case 4:
    return 0;
  case 5:
    return -1;
  case 6:
    return -w+1;
  case 7:
    return -w;
  case 8:
    return -w-1;
  }

  // must not come here
  assert(false);
}

template<typename T>
void matrix_shift_add(MatrixView<T, MatrixOrder::ColMajor>& buf,
                      MatrixView<T, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p,
                      const int block_offset) {
  Measurement::Start("matrix_shift_add1");

  const int h = p.input_height;
  const int w = p.input_width;
  const int oc = p.output_channels;
  const int kh = p.kernel_height;
  const int kw = p.kernel_width;
  const auto col_block = buf.cols();

  // only 3x3 kernel is supported.
  assert(kh == 3 && kw == 3);

  for (unsigned int j = 0; j < col_block; ++j) {
    for (unsigned int i = 0; i < buf.rows(); ++i) {
      if (is_first_column(j + block_offset, w) && is_cfi(i, p.output_channels)) {
        buf.set(i, j, 0);
      } else if (is_last_column(j + block_offset, w) && is_adg(i, p.output_channels)) {
        buf.set(i, j, 0);
      }
    }
  }

  Measurement::Stop();

  Measurement::Start("matrix_shift_add2");

  for (int k = 0; k < col_block; ++k) {
    const auto true_k = k + block_offset;
    for (unsigned int i = 0; i < kh * kw; ++i) {
      int offset = calc_offset(i, w);
      if ((true_k + offset < 0) || (true_k + offset >= h * w)) {
        continue;
      }

      T* r = result.data(0, true_k + offset);
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

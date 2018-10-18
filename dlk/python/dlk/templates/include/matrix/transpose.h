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

#ifndef DLK_MATRIX_TRANSPOSE_H_INCLUDED
#define DLK_MATRIX_TRANSPOSE_H_INCLUDED

#include "matrix_view.h"
#include "time_measurement.h"

namespace dlk {

template<typename T>
void matrix_transpose(MatrixView<T, MatrixOrder::ColMajor>& m, MatrixView<T, MatrixOrder::ColMajor>& out) {
  Measurement::Start("matrix_transpose (col_major)");

  for (unsigned int j = 0; j < m.cols(); ++j) {
    for (unsigned int i = 0; i < m.rows(); ++i) {
      *out.data(j,i) = *m.data(i,j);
    }
  }

  Measurement::Stop();
}

template<typename T>
void matrix_transpose(MatrixView<T, MatrixOrder::RowMajor>& m, MatrixView<T, MatrixOrder::RowMajor>& out) {
  Measurement::Start("matrix_transpose (row_major)");

  for (unsigned int i = 0; i < m.rows(); ++i) {
    for (unsigned int j = 0; j < m.cols(); ++j) {
      *out.data(j,i) = *m.data(i,j);
    }
  }

  Measurement::Stop();
}

} // namespace dlk

#endif // DLK_MATRIX_TRANSPOSE_H_INCLUDED

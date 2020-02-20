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

#ifndef DLK_MATRIX_COL_MAJOR_TO_ROW_MAJOR_H_INCLUDED
#define DLK_MATRIX_COL_MAJOR_TO_ROW_MAJOR_H_INCLUDED

#include "matrix_view.h"
#include "matrix/transpose.h"

namespace dlk {

// buf size must be larger than m.rows() * m.cols() * sizeof(T)
template<typename T>
MatrixView<T, MatrixOrder::RowMajor> col_major_to_row_major(MatrixView<T, MatrixOrder::ColMajor>& m, T* buf) {
   auto buf_mv = MatrixView<T, MatrixOrder::ColMajor>(buf, m.cols(), m.rows());
   matrix_transpose(m, buf_mv);

   return MatrixView<T, MatrixOrder::RowMajor>(buf, m.rows(), m.cols());
}

} // namespace dlk

#endif // DLK_MATRIX_COL_MAJOR_TO_ROW_MAJOR_H_INCLUDED

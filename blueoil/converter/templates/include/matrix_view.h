#ifndef DLK_MATRIX_H_INCLUDED
#define DLK_MATRIX_H_INCLUDED
#include <cassert>

// origin of this code is MatrixMap class in gemmlowp.
// this file follow the original license (Apache License, Version 2.0)

// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace dlk {

enum class MatrixOrder { ColMajor, RowMajor };

// A MatrixView is a view of an existing buffer as a matrix. It does not own
// the buffer.
template <typename tScalar, MatrixOrder tOrder> class MatrixView {
public:
  typedef tScalar Scalar;
  static const MatrixOrder kOrder = tOrder;

protected:
  Scalar *data_; // not owned.
  int rows_, cols_, stride_;

public:
  MatrixView() : data_(nullptr), rows_(0), cols_(0), stride_(0) {}
  MatrixView(Scalar *data, int rows, int cols)
      : data_(data), rows_(rows), cols_(cols),
        stride_(kOrder == MatrixOrder::ColMajor ? rows : cols) {}
  MatrixView(Scalar *data, int rows, int cols, int stride)
      : data_(data), rows_(rows), cols_(cols), stride_(stride) {}
  MatrixView(const MatrixView &other)
      : data_(other.data_), rows_(other.rows_), cols_(other.cols_),
        stride_(other.stride_) {}

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int stride() const { return stride_; }
  inline int rows_stride() const {
    return kOrder == MatrixOrder::ColMajor ? 1 : stride_;
  }
  inline int cols_stride() const {
    return kOrder == MatrixOrder::RowMajor ? 1 : stride_;
  }
  Scalar *data() const { return data_; }
  inline Scalar *data(int row, int col) const {
    return data_ + row * rows_stride() + col * cols_stride();
  }

  void set(int row, int col, Scalar v) const { *data(row, col) = v; }

  Scalar &operator()(int row, int col) const { return *data(row, col); }

  MatrixView block(int start_row, int start_col, int block_rows,
                   int block_cols) const {
    assert(start_row >= 0);
    assert(start_row + block_rows <= rows_);
    assert(start_col >= 0);
    assert(start_col + block_cols <= cols_);

    return MatrixView(data(start_row, start_col), block_rows, block_cols,
                      stride_);
  }
};

} // namespace dlk

#endif /* DLK_MATRIX_H_INCLUDED */

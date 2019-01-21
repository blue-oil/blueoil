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

#ifndef DLK_MATRIX_MULTIPLICATION_H_INCLUDED
#define DLK_MATRIX_MULTIPLICATION_H_INCLUDED

#include "matrix_view.h"
#include "matrix/row_major_to_col_major.h"
#include "time_measurement.h"

#if defined(USE_NEON) || defined(USE_ASIMD)
  #include <arm_neon.h>
#endif

namespace dlk {

namespace details {

inline void matrix_multiplication_col3(
  MatrixView<float, MatrixOrder::RowMajor>& A,
  MatrixView<float, MatrixOrder::ColMajor>& B,
  MatrixView<float, MatrixOrder::ColMajor>& C) {
#if defined(USE_NEON) || defined(USE_ASIMD)
  auto A_colm = row_major_to_col_major(A);
  for (std::size_t i = 0; i < B.cols(); ++i) {
    float32x4_t rhs0 = vdupq_n_f32((float)(*B.data(0, i)));
    float32x4_t rhs1 = vdupq_n_f32((float)(*B.data(1, i)));
    float32x4_t rhs2 = vdupq_n_f32((float)(*B.data(2, i)));

    assert(A.rows() % 4 == 0);
    for (std::size_t j = 0; j + 3 < A.rows(); j += 4) {
      float32x4_t lhs0 = vld1q_f32(A_colm.data(j, 0));
      float32x4_t lhs1 = vld1q_f32(A_colm.data(j, 1));
      float32x4_t lhs2 = vld1q_f32(A_colm.data(j, 2));

      float32x4_t r;
      r = vmulq_f32(lhs0, rhs0);
      r = vmlaq_f32(r, lhs1, rhs1);
      r = vmlaq_f32(r, lhs2, rhs2);
      vst1q_f32(C.data(j, i), r);
    }
  }
#endif
}

} // namespace details

// FIXME: this implementation is very slow...
template<typename T, typename U, typename V>
void matrix_multiplication(
   MatrixView<T, MatrixOrder::RowMajor>& A,
   MatrixView<U, MatrixOrder::ColMajor>& B,
   MatrixView<V, MatrixOrder::ColMajor>& C) {

  assert(A.cols() == B.rows());
  Measurement::Start("matrix_multiplication");

#if defined(USE_NEON) || defined(USE_ASIMD)
  if (A.cols() == 3 && A.rows() % 4 == 0) {
      details::matrix_multiplication_col3(A, B, C);
    Measurement::Stop();
    return;
  }
#endif

  constexpr unsigned int block_size_i = 16; // configurable, multiple of 4
  constexpr unsigned int block_size_j = 16; // configurable
  constexpr unsigned int block_size_k = 16; // configurable, multiple of 4
  static_assert((block_size_i % 4) == 0, "block_size_i must be multiple of 4");
  static_assert((block_size_k % 4) == 0, "block_size_k must be multiple of 4");
  for (unsigned int i = 0; i < A.rows(); i+=block_size_i) {
    for (unsigned int k = 0; k < B.cols(); k+=block_size_k) {
      V r[block_size_i][block_size_k] = {};
      for (unsigned int j = 0; j < A.cols(); j+=block_size_j) {
        if (A.rows() - i < block_size_i ||
          A.cols() - j < block_size_j ||
          B.cols() - k < block_size_k) {
          for (unsigned int i2 = 0; i2 < std::min(block_size_i, A.rows() - i); ++i2) {
            for (unsigned int k2 = 0; k2 < std::min(block_size_k, B.cols() - k); ++k2) {
              for (unsigned int j2 = 0; j2 < std::min(block_size_j, A.cols() - j); ++j2) {
                r[i2][k2] += *A.data(i+i2, j+j2) * *B.data(j+j2, k+k2);
              }
            }
          }
        } else {
          for (unsigned int i2 = 0; i2 < block_size_i; i2+=4) {
            for (unsigned int k2 = 0; k2 < block_size_k; k2+=4) {
              V r00 = 0;
              V r01 = 0;
              V r02 = 0;
              V r03 = 0;
              V r10 = 0;
              V r11 = 0;
              V r12 = 0;
              V r13 = 0;
              V r20 = 0;
              V r21 = 0;
              V r22 = 0;
              V r23 = 0;
              V r30 = 0;
              V r31 = 0;
              V r32 = 0;
              V r33 = 0;
              for (unsigned int j2 = 0; j2 < block_size_j; ++j2) {
                U b0 = *B.data(j+j2, k+k2+0);
                U b1 = *B.data(j+j2, k+k2+1);
                U b2 = *B.data(j+j2, k+k2+2);
                U b3 = *B.data(j+j2, k+k2+3);
                T a = *A.data(i+i2+0, j+j2);
                r00 += a * b0;
                r01 += a * b1;
                r02 += a * b2;
                r03 += a * b3;
                a = *A.data(i+i2+1, j+j2);
                r10 += a * b0;
                r11 += a * b1;
                r12 += a * b2;
                r13 += a * b3;
                a = *A.data(i+i2+2, j+j2);
                r20 += a * b0;
                r21 += a * b1;
                r22 += a * b2;
                r23 += a * b3;
                a = *A.data(i+i2+3, j+j2);
                r30 += a * b0;
                r31 += a * b1;
                r32 += a * b2;
                r33 += a * b3;
              }
              r[i2+0][k2+0] += r00;
              r[i2+0][k2+1] += r01;
              r[i2+0][k2+2] += r02;
              r[i2+0][k2+3] += r03;
              r[i2+1][k2+0] += r10;
              r[i2+1][k2+1] += r11;
              r[i2+1][k2+2] += r12;
              r[i2+1][k2+3] += r13;
              r[i2+2][k2+0] += r20;
              r[i2+2][k2+1] += r21;
              r[i2+2][k2+2] += r22;
              r[i2+2][k2+3] += r23;
              r[i2+3][k2+0] += r30;
              r[i2+3][k2+1] += r31;
              r[i2+3][k2+2] += r32;
              r[i2+3][k2+3] += r33;
            }
          }
        }
      }
      for (unsigned int k2 = 0; k2 < std::min(block_size_k, B.cols() - k); ++k2) {
        for (unsigned int i2 = 0; i2 < std::min(block_size_i, A.rows() - i); ++i2) {
          C.set(i+i2, k+k2, r[i2][k2]);
        }
      }
    }
  }

  Measurement::Stop();
}

} // namespace dlk

#endif // DLK_MATRIX_MULTIPLICATION_H_INCLUDED

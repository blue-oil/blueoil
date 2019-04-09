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

#include <algorithm>
#include <cassert>
#include <thread>
#include <vector>

#include "global.h"
#include "matrix_view.h"
#include "matrix/quantized_multiplication.h"
#include "time_measurement.h"

namespace {

static int pop_count(T_UINT i) {
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

#define CONV(i, k) r##i##k += pop_count(~(a ^ b##k##0)) + 2 * pop_count(~(a ^ b##k##1)) - 3 * (pop_count(~a));

void quantized_matrix_multiplication_body(
  const dlk::MatrixView<T_UINT, dlk::MatrixOrder::RowMajor>& A,
  const dlk::MatrixView<QUANTIZED_PACKED, dlk::MatrixOrder::ColMajor>& B,
  unsigned int begin,
  unsigned int end,
  dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor>& C) {
  constexpr unsigned int block_size_i = 4; // not configurable, hardcoded
  constexpr unsigned int block_size_j = 4; // configurable
  constexpr unsigned int block_size_k = 4; // not configurable, hardcoded
  static_assert(block_size_i == 4, "block_size_i must be 4");
  static_assert(block_size_k == 4, "block_size_k must be 4");
  for (unsigned int k = begin; k < end; k += block_size_k) {
    for (unsigned int i = 0; i < A.rows(); i += block_size_i) {
      BIN_CONV_OUTPUT r00 = 0;
      BIN_CONV_OUTPUT r01 = 0;
      BIN_CONV_OUTPUT r02 = 0;
      BIN_CONV_OUTPUT r03 = 0;
      BIN_CONV_OUTPUT r10 = 0;
      BIN_CONV_OUTPUT r11 = 0;
      BIN_CONV_OUTPUT r12 = 0;
      BIN_CONV_OUTPUT r13 = 0;
      BIN_CONV_OUTPUT r20 = 0;
      BIN_CONV_OUTPUT r21 = 0;
      BIN_CONV_OUTPUT r22 = 0;
      BIN_CONV_OUTPUT r23 = 0;
      BIN_CONV_OUTPUT r30 = 0;
      BIN_CONV_OUTPUT r31 = 0;
      BIN_CONV_OUTPUT r32 = 0;
      BIN_CONV_OUTPUT r33 = 0;
      for (unsigned int j = 0; j < A.cols(); j += block_size_j) {
        if (end - k < block_size_k ||
          A.rows() - i < block_size_i ||
          A.cols() - j < block_size_j) {
          for (unsigned int j2 = 0; j2 < std::min(block_size_j, A.cols() - j); ++j2) {
            QUANTIZED_PACKED b00;
            QUANTIZED_PACKED b01;
            QUANTIZED_PACKED b10;
            QUANTIZED_PACKED b11;
            QUANTIZED_PACKED b20;
            QUANTIZED_PACKED b21;
            QUANTIZED_PACKED b30;
            QUANTIZED_PACKED b31;
            if (end - k > 0) {
              b00 = *B.data(2*(j+j2)  , k+0);
              b01 = *B.data(2*(j+j2)+1, k+0);
            }
            if (end - k > 1) {
              b10 = *B.data(2*(j+j2)  , k+1);
              b11 = *B.data(2*(j+j2)+1, k+1);
            }
            if (end - k > 2) {
              b20 = *B.data(2*(j+j2)  , k+2);
              b21 = *B.data(2*(j+j2)+1, k+2);
            }
            if (end - k > 3) {
              b30 = *B.data(2*(j+j2)  , k+3);
              b31 = *B.data(2*(j+j2)+1, k+3);
            }
            if (A.rows() - i > 0) {
              const auto a = QUANTIZED_PACKED(*A.data(i+0, j+j2));
              if (end - k > 0) CONV(0, 0)
              if (end - k > 1) CONV(0, 1)
              if (end - k > 2) CONV(0, 2)
              if (end - k > 3) CONV(0, 3)
            }
            if (A.rows() - i > 0) {
              const auto a = QUANTIZED_PACKED(*A.data(i+1, j+j2));
              if (end - k > 0) CONV(1, 0)
              if (end - k > 1) CONV(1, 1)
              if (end - k > 2) CONV(1, 2)
              if (end - k > 3) CONV(1, 3)
            }
            if (A.rows() - i > 0) {
              const auto a = QUANTIZED_PACKED(*A.data(i+2, j+j2));
              if (end - k > 0) CONV(2, 0)
              if (end - k > 1) CONV(2, 1)
              if (end - k > 2) CONV(2, 2)
              if (end - k > 3) CONV(2, 3)
            }
            if (A.rows() - i > 0) {
              const auto a = QUANTIZED_PACKED(*A.data(i+3, j+j2));
              if (end - k > 0) CONV(3, 0)
              if (end - k > 1) CONV(3, 1)
              if (end - k > 2) CONV(3, 2)
              if (end - k > 3) CONV(3, 3)
            }
          }
        } else {
          for (unsigned int j2 = 0; j2 < block_size_j; ++j2) {
            QUANTIZED_PACKED b00 = *B.data(2*(j+j2)  , k+0);
            QUANTIZED_PACKED b01 = *B.data(2*(j+j2)+1, k+0);
            QUANTIZED_PACKED b10 = *B.data(2*(j+j2)  , k+1);
            QUANTIZED_PACKED b11 = *B.data(2*(j+j2)+1, k+1);
            QUANTIZED_PACKED b20 = *B.data(2*(j+j2)  , k+2);
            QUANTIZED_PACKED b21 = *B.data(2*(j+j2)+1, k+2);
            QUANTIZED_PACKED b30 = *B.data(2*(j+j2)  , k+3);
            QUANTIZED_PACKED b31 = *B.data(2*(j+j2)+1, k+3);
            auto a = QUANTIZED_PACKED(*A.data(i+0, j+j2));
            CONV(0, 0)
            CONV(0, 1)
            CONV(0, 2)
            CONV(0, 3)
            a = QUANTIZED_PACKED(*A.data(i+1, j+j2));
            CONV(1, 0)
            CONV(1, 1)
            CONV(1, 2)
            CONV(1, 3)
            a = QUANTIZED_PACKED(*A.data(i+2, j+j2));
            CONV(2, 0)
            CONV(2, 1)
            CONV(2, 2)
            CONV(2, 3)
            a = QUANTIZED_PACKED(*A.data(i+3, j+j2));
            CONV(3, 0)
            CONV(3, 1)
            CONV(3, 2)
            CONV(3, 3)
          }
        }
      }
      if (A.rows() - i > 0) {
        if (end - k > 0) C.set(i+0, k+0, r00);
        if (end - k > 1) C.set(i+0, k+1, r01);
        if (end - k > 2) C.set(i+0, k+2, r02);
        if (end - k > 3) C.set(i+0, k+3, r03);
      }
      if (A.rows() - i > 1) {
        if (end - k > 0) C.set(i+1, k+0, r10);
        if (end - k > 1) C.set(i+1, k+1, r11);
        if (end - k > 2) C.set(i+1, k+2, r12);
        if (end - k > 3) C.set(i+1, k+3, r13);
      }
      if (A.rows() - i > 2) {
        if (end - k > 0) C.set(i+2, k+0, r20);
        if (end - k > 1) C.set(i+2, k+1, r21);
        if (end - k > 2) C.set(i+2, k+2, r22);
        if (end - k > 3) C.set(i+2, k+3, r23);
      }
      if (A.rows() - i > 3) {
        if (end - k > 0) C.set(i+3, k+0, r30);
        if (end - k > 1) C.set(i+3, k+1, r31);
        if (end - k > 2) C.set(i+3, k+2, r32);
        if (end - k > 3) C.set(i+3, k+3, r33);
      }
    }
  }
}

#undef CONV

} // namespace

namespace dlk {

void quantized_matrix_multiplication(
  const MatrixView<T_UINT, MatrixOrder::RowMajor>& A,
  const MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>& B,
  MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>& C) {
  Measurement::Start("quantized_matrix_multiplication");

  assert(A.cols() * 2 == B.rows());

  unsigned int chunk_size = B.cols() / std::thread::hardware_concurrency();
  if (chunk_size == 0) {
    chunk_size += 1;
  }

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < B.cols(); i += chunk_size) {
    threads.emplace_back(std::thread([A, B, &C, i, chunk_size] {
          quantized_matrix_multiplication_body(A, B, i, std::min(i + chunk_size, static_cast<unsigned int>(B.cols())), C);
    }));
  }

  for (auto& th: threads) {
    th.join();
  }

  Measurement::Stop();
}

} // namespace dlk

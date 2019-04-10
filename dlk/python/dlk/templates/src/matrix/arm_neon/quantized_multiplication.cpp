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

#include <arm_neon.h>

namespace {

static int16x4_t pop_count_4(uint32x4_t v) {
  uint8x16_t u = vreinterpretq_u8_u32(v);
  uint8x16_t c = vcntq_u8(u);
  uint32x4_t r = vreinterpretq_u32_u8(c);
  uint32x4_t m = vmulq_u32(r, vdupq_n_u32(0x01010101));
  uint32x4_t s = vshrq_n_u32(m, 24);
  return vqmovn_s32(vreinterpretq_s32_u32(s));
}

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
      int16x4_t res0 = vdup_n_s16(0);
      int16x4_t res1 = vdup_n_s16(0);
      int16x4_t res2 = vdup_n_s16(0);
      int16x4_t res3 = vdup_n_s16(0);
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
            QUANTIZED_PACKED b_ary[4][2] = {
              {b00, b01},
              {b10, b11},
              {b20, b21},
              {b30, b31}
            };
            uint32x4x2_t b = vld2q_u32((uint32_t*)&b_ary[0][0]);
            if (A.rows() - i > 0) {
              uint32x4_t a = vdupq_n_u32(*A.data(i+0, j+j2));
              res0 += pop_count_4(~(a ^ b.val[0]))
                + (pop_count_4(~(a ^ b.val[1])) << 1)
                - 3 * pop_count_4(~a);
            }
            if (A.rows() - i > 1) {
              uint32x4_t a = vdupq_n_u32(*A.data(i+1, j+j2));
              res1 += pop_count_4(~(a ^ b.val[0]))
                + (pop_count_4(~(a ^ b.val[1])) << 1)
                - 3 * pop_count_4(~a);
            }
            if (A.rows() - i > 2) {
              uint32x4_t a = vdupq_n_u32(*A.data(i+2, j+j2));
              res2 += pop_count_4(~(a ^ b.val[0]))
                + (pop_count_4(~(a ^ b.val[1])) << 1)
                - 3 * pop_count_4(~a);
            }
            if (A.rows() - i > 3) {
              uint32x4_t a = vdupq_n_u32(*A.data(i+3, j+j2));
              res3 += pop_count_4(~(a ^ b.val[0]))
                + (pop_count_4(~(a ^ b.val[1])) << 1)
                - 3 * pop_count_4(~a);
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
            QUANTIZED_PACKED b_ary[4][2] = {
              {b00, b01},
              {b10, b11},
              {b20, b21},
              {b30, b31}
            };
            uint32x4x2_t b = vld2q_u32((uint32_t*)&b_ary[0][0]);
            uint32x4_t a = vdupq_n_u32(*A.data(i+0, j+j2));
            res0 += pop_count_4(~(a ^ b.val[0]))
              + (pop_count_4(~(a ^ b.val[1])) << 1)
              - 3 * pop_count_4(~a);
            a = vdupq_n_u32(*A.data(i+1, j+j2));
            res1 += pop_count_4(~(a ^ b.val[0]))
              + (pop_count_4(~(a ^ b.val[1])) << 1)
              - 3 * pop_count_4(~a);
            a = vdupq_n_u32(*A.data(i+2, j+j2));
            res2 += pop_count_4(~(a ^ b.val[0]))
              + (pop_count_4(~(a ^ b.val[1])) << 1)
              - 3 * pop_count_4(~a);
            a = vdupq_n_u32(*A.data(i+3, j+j2));
            res3 += pop_count_4(~(a ^ b.val[0]))
              + (pop_count_4(~(a ^ b.val[1])) << 1)
              - 3 * pop_count_4(~a);
          }
        }
      }
      if (A.rows() - i > 0) {
        if (end - k > 0) C.set(i+0, k+0, vget_lane_s16(res0, 0));
        if (end - k > 1) C.set(i+0, k+1, vget_lane_s16(res0, 1));
        if (end - k > 2) C.set(i+0, k+2, vget_lane_s16(res0, 2));
        if (end - k > 3) C.set(i+0, k+3, vget_lane_s16(res0, 3));
      }
      if (A.rows() - i > 1) {
        if (end - k > 0) C.set(i+1, k+0, vget_lane_s16(res1, 0));
        if (end - k > 1) C.set(i+1, k+1, vget_lane_s16(res1, 1));
        if (end - k > 2) C.set(i+1, k+2, vget_lane_s16(res1, 2));
        if (end - k > 3) C.set(i+1, k+3, vget_lane_s16(res1, 3));
      }
      if (A.rows() - i > 2) {
        if (end - k > 0) C.set(i+2, k+0, vget_lane_s16(res2, 0));
        if (end - k > 1) C.set(i+2, k+1, vget_lane_s16(res2, 1));
        if (end - k > 2) C.set(i+2, k+2, vget_lane_s16(res2, 2));
        if (end - k > 3) C.set(i+2, k+3, vget_lane_s16(res2, 3));
      }
      if (A.rows() - i > 3) {
        if (end - k > 0) C.set(i+3, k+0, vget_lane_s16(res3, 0));
        if (end - k > 1) C.set(i+3, k+1, vget_lane_s16(res3, 1));
        if (end - k > 2) C.set(i+3, k+2, vget_lane_s16(res3, 2));
        if (end - k > 3) C.set(i+3, k+3, vget_lane_s16(res3, 3));
      }
    }
  }
}

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

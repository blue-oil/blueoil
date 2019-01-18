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

#include <thread>
#include <vector>

#include "global.h"
#include "matrix_view.h"
#include "matrix/shift_add.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

#include <arm_neon.h>

namespace dlk {

template<>
void matrix_shift_add(MatrixView<float, MatrixOrder::ColMajor>& buf,
                      MatrixView<float, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p) {
  Measurement::Start("matrix_shift_add_f1");

  const int h = p.input_height;
  const int w = p.input_width;
  const int oc = p.output_channels;
  const int kh = p.kernel_height;
  const int kw = p.kernel_width;

  // only 3x3 kernel is supported.
  assert(kh == 3 && kw == 3);

  for (unsigned int j = 0; j < buf.cols(); ++j) {
    for (unsigned int i = 0; i < buf.rows(); ++i) {
      if (is_first_column(j, w) && is_cfi(i, p.output_channels)) {
        buf.set(i, j, 0);
      } else if (is_last_column(j, w) && is_adg(i, p.output_channels)) {
        buf.set(i, j, 0);
      }
    }
  }

  Measurement::Stop();

  Measurement::Start("matrix_shift_add_f2");

  unsigned int chunk_size = (h * w) / static_cast<unsigned int>(std::thread::hardware_concurrency());
  if (chunk_size == 0) {
    chunk_size += 1;
  }

  std::vector<std::thread> threads;
  for (unsigned int begin = 0; begin < (h * w); begin += chunk_size) {
    threads.emplace_back(std::thread([buf, begin, chunk_size, h, w, p, &result] {
          const int kh = p.kernel_height;
          const int kw = p.kernel_width;
          const int oc = p.output_channels;
          for (int k = begin; k < std::min(begin + chunk_size, static_cast<unsigned int>(h * w)); ++k) {
            for (unsigned int i = 0; i < (kh * kw); ++i) {
              int offset = calc_offset(i, w);
              if ((k - offset < 0) || (k - offset >= (h * w))) {
                continue;
              }

              float* r = result.data(0, k);
              float* b = buf.data(i*oc, k - offset);


              unsigned int j = 0;
              for (; j + 3 < oc; j += 4) {
                float32x4_t b_ = vld1q_f32(b+j);
                float32x4_t r_ = vld1q_f32(r+j);
                float32x4_t r__ = vaddq_f32(b_, r_);
                vst1q_f32(r+j, r__);
              }

              for (; j < oc; ++j) {
                r[j] += b[j];
              }
            }
          }
        }));
  }

  for (auto& th: threads) {
    if (th.joinable())
      th.join();
  }

  Measurement::Stop();
}

template<>
void matrix_shift_add(MatrixView<int32_t, MatrixOrder::ColMajor>& buf,
                      MatrixView<int32_t, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p) {
  Measurement::Start("matrix_shift_add_i1");

  const int h = p.input_height;
  const int w = p.input_width;
  const int oc = p.output_channels;
  const int kh = p.kernel_height;
  const int kw = p.kernel_width;

  // only 3x3 kernel is supported.
  assert(kh == 3 && kw == 3);

  for (unsigned int j = 0; j < buf.cols(); ++j) {
    for (unsigned int i = 0; i < buf.rows(); ++i) {
      if (is_first_column(j, w) && is_cfi(i, p.output_channels)) {
        buf.set(i, j, 0);
      } else if (is_last_column(j, w) && is_adg(i, p.output_channels)) {
        buf.set(i, j, 0);
      }
    }
  }

  Measurement::Stop();

  Measurement::Start("matrix_shift_add_i2");

  unsigned int chunk_size = (h * w) / static_cast<unsigned int>(std::thread::hardware_concurrency());
  if (chunk_size == 0) {
    chunk_size += 1;
  }

  std::vector<std::thread> threads;
  for (unsigned int begin = 0; begin < h * w; begin += chunk_size) {
    threads.emplace_back(std::thread([buf, begin, chunk_size, h, w, p, &result] {
          const int kh = p.kernel_height;
          const int kw = p.kernel_width;
          const int oc = p.output_channels;
          for (int k = begin; k < std::min(begin + chunk_size, static_cast<unsigned int>(h * w)); ++k) {
            for (unsigned int i = 0; i < kh * kw; ++i) {
              int offset = calc_offset(i, w);
              if ((k - offset < 0) || (k - offset >= h * w)) {
                continue;
              }

              int32_t* r = result.data(0, k);
              int32_t* b = buf.data(i*oc, k - offset);


              unsigned int j = 0;
              for (; j + 3 < oc; j += 4) {
                int32x4_t b_ = vld1q_s32(b+j);
                int32x4_t r_ = vld1q_s32(r+j);
                int32x4_t r__ = vaddq_s32(b_, r_);
                vst1q_s32(r+j, r__);
              }

              for (; j < oc; ++j) {
                r[j] += b[j];
              }
            }
          }
    }));
  }

  for (auto& th: threads) {
    if (th.joinable())
      th.join();
  }

  Measurement::Stop();
}

} // namespace dlk

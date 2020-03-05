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

#include "global.h"
#include "matrix_view.h"
#include "matrix/shift_add.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"

#ifdef USE_NEON
  #include <arm_neon.h>
#elif defined USE_AVX
  #include <x86intrin.h>
#endif

namespace dlk {

template<>
void matrix_shift_add(MatrixView<float, MatrixOrder::ColMajor>& buf,
                      MatrixView<float, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p,
                      const int block_offset) {
  Measurement::Start("matrix_shift_add_f");

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

  const auto res_col_start = std::max<std::ptrdiff_t>(0, block_offset - pad * w - pad);
  const auto res_col_end = std::min(h * w, block_offset + col_block + pad * w + pad);
#pragma omp parallel for
  for (int k = res_col_start; k < res_col_end; ++k) {
    const auto row = k / w;
    const auto col = k % w;
    for (int kr = 0; kr < kh; ++kr) {
      for (int kc = 0; kc < kw; ++kc) {
        if (row + kr < pad || row + kr >= h + pad || col + kc < pad || col + kc >= w + pad) continue;
        const auto offset = (kr - pad) * w + (kc - pad);
        const auto b_col = k - block_offset + offset;
        if (b_col < 0 || col_block <= b_col) continue;

        float* r = result.data(0, k);
        float* b = buf.data((kr*kw + kc)*oc, b_col);


        unsigned int j = 0;
#ifdef USE_NEON
        for (; j + 3 < oc; j += 4) {
          float32x4_t b_ = vld1q_f32(b+j);
          float32x4_t r_ = vld1q_f32(r+j);
          float32x4_t r__ = vaddq_f32(b_, r_);
          vst1q_f32(r+j, r__);
        }
#elif defined USE_AVX
        for (; j + 7 < oc; j += 8) {
          auto vb = _mm256_loadu_ps(b+j);
          auto vr = _mm256_loadu_ps(r+j);
          auto res = _mm256_add_ps(vr, vb);
          _mm256_storeu_ps(r+j, res);
        }
#endif
        for (; j < oc; ++j) {
          r[j] += b[j];
        }
      }
    }
  }

  Measurement::Stop();
}

template<>
void matrix_shift_add(MatrixView<int32_t, MatrixOrder::ColMajor>& buf,
                      MatrixView<int32_t, MatrixOrder::ColMajor>& result,
                      const struct convolution_parameters& p,
                      const int block_offset) {
  Measurement::Start("matrix_shift_add_i");

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

  const auto res_col_start = std::max<std::ptrdiff_t>(0, block_offset - pad * w - pad);
  const auto res_col_end = std::min(h * w, block_offset + col_block + pad * w + pad);
#pragma omp parallel for
  for (int k = res_col_start; k < res_col_end; ++k) {
    const auto row = k / w;
    const auto col = k % w;
    for (int kr = 0; kr < kh; ++kr) {
      for (int kc = 0; kc < kw; ++kc) {
        if (row + kr < pad || row + kr >= h + pad || col + kc < pad || col + kc >= w + pad) continue;
        const auto offset = (kr - pad) * w + (kc - pad);
        const auto b_col = k - block_offset + offset;
        if (b_col < 0 || col_block <= b_col) continue;

        int32_t* r = result.data(0, k);
        int32_t* b = buf.data((kr*kw + kc)*oc, b_col);


        unsigned int j = 0;
#ifdef USE_NEON
        for (; j + 3 < oc; j += 4) {
          int32x4_t b_ = vld1q_s32(b+j);
          int32x4_t r_ = vld1q_s32(r+j);
          int32x4_t r__ = vaddq_s32(b_, r_);
          vst1q_s32(r+j, r__);
        }
#elif defined USE_AVX
        for (; j + 7 < oc; j += 8) {
          auto vb = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b+j));
          auto vr = _mm256_loadu_si256(reinterpret_cast<__m256i*>(r+j));
          auto res = _mm256_add_epi32(vr, vb);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(r+j), res);
        }
#endif
        for (; j < oc; ++j) {
          r[j] += b[j];
        }
      }
    }
  }

  Measurement::Stop();
}

} // namespace dlk

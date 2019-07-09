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
#include "operators.h" // FIXME(nikolay): for binary_convolution_parameters definition, rid of it later
#include "time_measurement.h"

#include <x86intrin.h>

namespace dlk {

namespace impl {

void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");

  const auto buf_ts0 = std::make_unique<BIN_CONV_OUTPUT[]>(result.rows());
  const auto buf_ts1 = std::make_unique<BIN_CONV_OUTPUT[]>(result.rows());
  const auto buf_ts2 = std::make_unique<BIN_CONV_OUTPUT[]>(result.rows());
  const auto buf_flg = std::make_unique<BIN_CONV_OUTPUT[]>(result.rows());

  for (unsigned int i = 0; i < result.rows(); ++i) {
    T_INT ts0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i];
    T_INT ts1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
    T_INT ts2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
    T_INT flag = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flag == -1) {
      ++ts0;
      ++ts1;
      ++ts2;
    }
    buf_ts0[i] = ts0;
    buf_ts1[i] = ts1;
    buf_ts2[i] = ts2;
    buf_flg[i] = flag;
  }

#pragma omp parallel for
  for (unsigned int j = 0; j < result.cols(); ++j) {
    for (unsigned int i = 0; i < result.rows(); i += 16) {
      const auto d = _mm256_loadu_si256(reinterpret_cast<__m256i*>(result.data(i, j)));
      const auto ts0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts0.get() + i));
      const auto ts1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts1.get() + i));
      const auto ts2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts2.get() + i));
      const auto flg = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_flg.get() + i));
      const auto f0 = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts0, d), flg);
      const auto f1 = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts1, d), flg);
      const auto f2 = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts2, d), flg);
      const auto is_neg = _mm256_cmpgt_epi16(_mm256_setzero_si256(), flg);
      const auto tmp = _mm256_add_epi16(_mm256_add_epi16(f0, f1), _mm256_add_epi16(f2, is_neg));
      const auto m2 = _mm256_sub_epi16(flg, _mm256_set1_epi16(2));
      const auto is_not_const = _mm256_cmpgt_epi16(_mm256_setzero_si256(), m2);
      const auto res = _mm256_blendv_epi8(m2, tmp, is_not_const);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data(i, j)), res);
    }
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

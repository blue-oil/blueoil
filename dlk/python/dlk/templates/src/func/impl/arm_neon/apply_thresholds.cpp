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

#include <memory>
#include <arm_neon.h>

namespace dlk {

namespace impl {

void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");

  const auto th = std::make_unique<BIN_CONV_OUTPUT[]>(NUM_OF_A2W1_THRESHOLD * result.rows());
  for (unsigned int i = 0; i < result.rows(); ++i) {
    auto th0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 0];
    auto th1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
    auto th2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
    const auto flg = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flg == -1) {
      ++th0;
      ++th1;
      ++th2;
    }
    th[NUM_OF_A2W1_THRESHOLD * i + 0] = th0;
    th[NUM_OF_A2W1_THRESHOLD * i + 1] = th1;
    th[NUM_OF_A2W1_THRESHOLD * i + 2] = th2;
    th[NUM_OF_A2W1_THRESHOLD * i + 3] = flg;
  }

#pragma omp parallel for
  for (unsigned int j = 0; j < result.cols(); ++j) {
    for (unsigned int i = 0; i < result.rows(); i += 8) {
      const auto d = vld1q_s16(result.data(i, j));
      const auto ts = vld4q_s16(th.get() + NUM_OF_A2W1_THRESHOLD * i);
      const auto f0 = vreinterpretq_s16_u16(vcgeq_s16(d, ts.val[0])) & ts.val[3];
      const auto f1 = vreinterpretq_s16_u16(vcgeq_s16(d, ts.val[1])) & ts.val[3];
      const auto f2 = vreinterpretq_s16_u16(vcgeq_s16(d, ts.val[2])) & ts.val[3];
      const auto is_neg = vreinterpretq_s16_u16(vcltq_s16(ts.val[3], vdupq_n_s16(0)));
      const auto tmp = f0 + f1 + f2 + is_neg;
      const auto m2 = vsubq_s16(ts.val[3], vdupq_n_s16(2));
      const auto is_const = vcgeq_s16(m2, vdupq_n_s16(0));
      const auto res = vbslq_s16(is_const, m2, tmp);
      vst1q_s16(result.data(i, j), res);
    }
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

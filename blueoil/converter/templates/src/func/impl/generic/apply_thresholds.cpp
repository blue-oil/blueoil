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

namespace dlk {

namespace impl {

void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");

#pragma omp parallel for
  for (unsigned int j = 0; j < result.cols(); ++j) {
    for (unsigned int i = 0; i < result.rows(); ++i) {
      BIN_CONV_OUTPUT d = *result.data(i, j);
      T_INT ts0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i];
      T_INT ts1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
      T_INT ts2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
      T_INT flag = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
      BIN_CONV_OUTPUT new_d;

      if (flag == 1) { // increasing function
        if (d < ts0)
          new_d = 0;
        else if (d < ts1)
          new_d = 1;
        else if (d < ts2)
          new_d = 2;
        else
          new_d = 3;
      } else if (flag == -1) { // decreasing function
        if (d > ts0)
          new_d = 0;
        else if (d > ts1)
          new_d = 1;
        else if (d > ts2)
          new_d = 2;
        else
          new_d = 3;
      } else if (flag == 0) { // ignore
        new_d = 0;
      } else {                            // constant function
        new_d = flag - 2;                 // note: 2 is a magic number!
        assert(0 <= new_d && new_d <= 3); // unsinged 2bits
      }
      *result.data(i, j) = new_d;
    }
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

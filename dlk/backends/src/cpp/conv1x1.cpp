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

#include "common/global.h"
#include "cpp/utils.h"

namespace cpp {
namespace p = conv1x1_params;

void conv1x1_impl(T_in in_data[], T_out out_data[], T_k k_data[], T_out threshold_data[], unsigned in_w, unsigned in_h,
                  unsigned in_c, unsigned out_c)
{
  unsigned idx_k = 0;
  unsigned idx_in = 0;
  unsigned idx_out = 0;
  unsigned idx_t = 0;

  T_k k_local[in_c][out_c];
  T_out threshold_local[p::out_c][p::num_thresholds];

  for (int oc = 0; oc < out_c; oc++) {
    for (unsigned ic = 0; ic < in_c; ic++) { k_local[ic][oc] = k_data[idx_k++]; }
  }

  if (threshold_data != NULL) {
    for (unsigned oc = 0; oc < p::out_c; oc++) {
      for (unsigned i = 0; i < p::num_thresholds; i++) { threshold_local[oc][i] = threshold_data[idx_t++]; }
    }
  }

  for (unsigned ih = 0; ih < in_h; ++ih)
    for (unsigned iw = 0; iw < in_w; ++iw) {
      T_out out[out_c];

      for (int oc = 0; oc < out_c; oc++) { out[oc] = 0; }

      for (unsigned ic = 0; ic < in_c; ic++) {
        T_in in_buf = in_data[idx_in++];

        for (int oc = 0; oc < out_c; oc++) {
          T_q k_buf = k_local[ic][oc];
          out[oc] += in_buf * k_buf;
        }
      }

      for (int oc = 0; oc < out_c; oc++) {
        T_out conv_result = out[oc];
        T_out out_buf;

        if (threshold_data != NULL) {
          T_out ts0 = threshold_local[oc][0];
          T_out ts1 = threshold_local[oc][1];
          T_out ts2 = threshold_local[oc][2];
          T_out flag = threshold_local[oc][3];

          if (flag == 1) // increasing function
          {
            if (conv_result < ts0)
              out_buf = 0;
            else if (conv_result < ts1)
              out_buf = 1;
            else if (conv_result < ts2)
              out_buf = 2;
            else
              out_buf = 3;
          } else if (flag == -1) // decreasing function
          {
            if (conv_result > ts2)
              out_buf = 0;
            else if (conv_result > ts1)
              out_buf = 1;
            else if (conv_result > ts0)
              out_buf = 2;
            else
              out_buf = 3;
          } else {
            // max value of 2 bits
            T_out k = 3 * 3 * out_c * 3;
            out_buf = flag - k;
          }
        } else {
          out_buf = conv_result;
        }

        out_data[idx_out++] = out_buf;
      }
    } // for LOOP_CONV_INPUT
}

void qconv1x1_impl(T_q in_data[], T_out out_data[], T_q k_data[], unsigned in_w, unsigned in_h, unsigned in_c_by_word,
                   unsigned out_c)
{
  unsigned idx_k = 0;
  unsigned idx_in = 0;
  unsigned idx_out = 0;

  T_k k_local[in_c_by_word][out_c];
  for (int oc = 0; oc < out_c; oc++) {
    for (unsigned ic = 0; ic < in_c_by_word; ic++) { k_local[ic][oc] = k_data[idx_k++]; }
  }

  for (unsigned ih = 0; ih < in_h; ++ih)
    for (unsigned iw = 0; iw < in_w; ++iw) {
      T_out out[out_c];

      for (int oc = 0; oc < out_c; oc++) { out[oc] = 0; }

      for (int ic = 0; ic < in_c_by_word; ic++) {
        T_q in_buf0 = in_data[idx_in++];
        T_q in_buf1 = in_data[idx_in++];

        for (int oc = 0; oc < out_c; oc++) {
          T_q k_buf = k_local[ic][oc];
          out[oc] += PE(k_buf, in_buf0, in_buf1);
        }
        idx_k++;
      }

      for (int oc = 0; oc < out_c; oc++) { out_data[idx_out + oc] = out[oc]; }
      idx_out += out_c;
    } // for LOOP_CONV_INPUT
}

} // namespace cpp

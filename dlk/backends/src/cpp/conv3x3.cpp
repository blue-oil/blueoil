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

#include <iostream>
#include "common/global.h"
#include "cpp/utils.h"

namespace cpp {
namespace p = conv3x3_params;

void conv3x3_impl(T_in in_data[], T_out out_data[], T_k k_data[], T_out threshold_data[], unsigned in_w, unsigned in_h,
                  unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c, unsigned pad, unsigned stride)
{
  T_k* k_local = new T_k[p::k_size * p::k_n];
  T_out threshold_local[p::out_c][p::num_thresholds];

  unsigned idx_k = 0;
  unsigned idx_t = 0;

  for (unsigned kn = 0; kn < p::k_n; kn++) {
    for (unsigned k = 0; k < p::k_size; k++) { k_local[k * p::k_n + kn] = k_data[idx_k++]; }
  }

  if (threshold_data != NULL) {
    for (unsigned oc = 0; oc < p::out_c; oc++) {
      for (unsigned i = 0; i < p::num_thresholds; i++) { threshold_local[oc][i] = threshold_data[idx_t++]; }
    }
  }

  unsigned idx_out = 0;

  for (unsigned oh = 0; oh < out_h; ++oh)
    for (unsigned ow = 0; ow < out_w; ++ow) {
      T_out out[p::k_n] = {};
      unsigned idx_k_local = 0;

      for (unsigned kh = 0; kh < p::k_h; kh++) {
        for (unsigned kw = 0; kw < p::k_w; kw++) {
          int ih = (oh * stride) - pad + kh;
          int iw = (ow * stride) - pad + kw;
          bool valid = (iw >= 0) && (iw < int(in_w)) && (ih >= 0) && (ih < int(in_h));

          for (unsigned kc = 0; kc < p::in_c; kc++) {
            if (valid) {
              int idx_in = ih * in_w * in_c + iw * in_c + kc;
              T_in in_buf = in_data[idx_in];

              for (int kn = 0; kn < p::k_n; kn++) {
                T_k k_buf = k_local[idx_k_local * p::k_n + kn];
                out[kn] += in_buf * k_buf;
              }
            }
            idx_k_local++;
          }
        }
      }

      for (int oc = 0; oc < p::out_c; oc++) {
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
            out_buf = flag - 2; // note: 2 is a magic number!
          }
        } else {
          out_buf = conv_result;
        }

        out_data[idx_out++] = out_buf;
      }

    } // for LOOP_CONV_INPUT

  delete[] k_local;
}

void qconv3x3_impl(T_q in_data[], T_out out_data[], T_q k_data[], T_out threshold_data, unsigned in_w, unsigned in_h,
                   unsigned in_c_by_word, unsigned out_w, unsigned out_h, unsigned out_c, unsigned pad, unsigned stride)
{
  unsigned idx_k = 0;

  for (int oc_out = 0; oc_out < out_c; oc_out += p::num_pe) {
    T_q* k_local = new T_q[p::k_size_packed * p::num_pe];

    for (unsigned kn = 0; kn < p::num_pe; ++kn) {
      for (unsigned k = 0; k < p::k_size_packed; ++k) { k_local[k * p::num_pe + kn] = k_data[idx_k++]; }
    }

    unsigned idx_out = oc_out;

    for (unsigned oh = 0; oh < out_h; ++oh)
      for (unsigned ow = 0; ow < out_w; ++ow) {
        T_out out[p::num_pe] = {};
        unsigned idx_k_local = 0;

        for (unsigned kh = 0; kh < p::k_h; ++kh)
          for (unsigned kw = 0; kw < p::k_w; ++kw)
            for (unsigned ic = 0; ic < in_c_by_word; ++ic) {
              int ih = (oh * stride) - pad + kh;
              int iw = (ow * stride) - pad + kw;
              bool valid = (iw >= 0) && (iw < int(in_w)) && (ih >= 0) && (ih < int(in_h));

              if (valid) {
                int idx_in = (ih * in_w * in_c_by_word + iw * in_c_by_word + ic) * p::nbits_in_data;
                T_q in_buf0 = in_data[idx_in];
                T_q in_buf1 = in_data[idx_in + 1];

                for (int kn = 0; kn < p::num_pe; kn++) {
                  T_q k_buf = k_local[idx_k_local * p::num_pe + kn];
                  out[kn] += PE(k_buf, in_buf0, in_buf1);
                }
              }
              idx_k_local++; // should be executed, even while no corresponding input
            }

        for (int kn = 0; kn < p::num_pe; kn++) { out_data[idx_out + kn] = out[kn]; }
        idx_out += out_c;
      } // for LOOP_CONV_INPUT

    delete[] k_local;
  }
}

} // namespace cpp

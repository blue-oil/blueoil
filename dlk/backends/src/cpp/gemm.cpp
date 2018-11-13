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

#define NBITS(T) (sizeof(T) * 8)

namespace cpp {
using namespace type;

template <class Ta, class Tb, class Ty>
void gemm(Ta A[], Tb B[], Ty Y[], u32 a_row, u32 a_col, u32 b_col)
{
  const unsigned y_row = a_row;
  const unsigned y_col = b_col;

  for (unsigned yr = 0; yr < y_row; ++yr)
    for (unsigned yc = 0; yc < y_col; ++yc) {
      Ty buf = 0;

      for (unsigned ac = 0; ac < a_col; ac++) {
        Ta a = A[yr * a_col + ac];
        Tb b = B[yc * a_col + ac];
        buf += a * b;
      }

      Y[yr * y_col + yc] = buf;
    }
}
template void gemm<i32, i32, i32>(i32[], i32[], i32[], u32, u32, u32);
template void gemm<u32, i32, i32>(u32[], i32[], i32[], u32, u32, u32);

template <class Tx, class Ty, int Nbit_a, int Nbit_b>
void qgemm(Tx A_packed[], Tx B_packed[], Ty Y[], u32 a_row, u32 a_col_by_word, u32 b_col)
{
  const unsigned y_row = a_row;
  const unsigned y_col = b_col;

  for (unsigned yr = 0; yr < y_row; ++yr)
    for (unsigned yc = 0; yc < y_col; ++yc) {
      Ty buf = 0;

      for (unsigned ac = 0; ac < a_col_by_word; ++ac) {
        Tx buf_a0 = A_packed[2 * (yr * a_col_by_word + ac)];
        Tx buf_a1 = A_packed[2 * (yr * a_col_by_word + ac) + 1];
        Tx buf_b = B_packed[yc * a_col_by_word + ac];

        buf += PE(buf_b, buf_a0, buf_a1);
      }

      Y[yr * y_col + yc] = buf;
    }
}
template void qgemm<u32, i32, 2, 1>(u32[], u32[], i32[], u32, u32, u32);

void qconv_with_kn2row_impl(T_q in_data[], T_out out_data[], T_q k_data[], T_out partial_out_data[],
                            T_out threshold_data[], unsigned in_w, unsigned in_h, unsigned in_c_by_word, unsigned out_w,
                            unsigned out_h, unsigned out_c, unsigned k_w, unsigned k_h, unsigned pad)
{
  static const unsigned num_pe = conv_common_params::num_pe;
  static const unsigned nbits_in_data = conv_common_params::nbits_in_data;
  static const unsigned num_thresholds = conv_common_params::num_thresholds;

  T_out threshold_local[out_c][num_thresholds];

  unsigned idx_k = 0;

  for (int8_t kh = 0; kh < k_h; kh++) {
    for (int8_t kw = 0; kw < k_w; kw++) {
      unsigned idx_t = 0;
      for (int16_t oc = 0; oc < out_c; oc += num_pe) {
        T_q* k_local = new T_q[in_c_by_word * num_pe];

        for (uint16_t kc = 0; kc < in_c_by_word; kc++) {
          for (uint16_t kn = 0; kn < num_pe; kn++) { k_local[kc * num_pe + kn] = k_data[idx_k++]; }
        }

        if (threshold_data != NULL) {
          for (unsigned kn = 0; kn < num_pe; kn++) {
            for (unsigned i = 0; i < num_thresholds; i++) { threshold_local[kn][i] = threshold_data[idx_t++]; }
          }
        }

        unsigned idx_out = oc;
        unsigned idx_in = 0;

        for (int16_t _ih = 0; _ih < in_h + 2 * pad; _ih++) {
          for (int16_t _iw = 0; _iw < in_w + 2 * pad; _iw++) {
            int16_t ih = _ih - pad;
            int16_t iw = _iw - pad;
            int16_t oh = _ih - kh;
            int16_t ow = _iw - kw;

            bool first_load = ((kh == 0) && (kw == 0));
            bool input_on = ((ih >= 0) && (ih < in_h) && (iw >= 0) && (iw < in_w));
            bool output_on = ((oh >= 0) && (oh < out_h) && (ow >= 0) && (ow < out_w));

            T_out out0[num_pe];
            T_out out1[num_pe];

            for (uint16_t kn = 0; kn < num_pe; kn++) {
              out0[kn] = (output_on && !first_load) ? partial_out_data[idx_out + kn] : 0;
            }

            for (uint16_t ic = 0; ic < in_c_by_word; ic++) {
              for (uint16_t ib = 0; ib < nbits_in_data; ib++) {
                T_q in = 0;

                if (input_on) {
                  in = in_data[idx_in++];
                }

                for (uint16_t kn = 0; kn < num_pe; kn++) {
                  T_q k = k_local[ic * num_pe + kn];
                  T_q nk_pop = pop_count(~k);
                  T_q xnor_pop = pop_count(~(in ^ k));
                  out0[kn] += ((xnor_pop - nk_pop) << ib);
                }
              }
            }

            for (uint16_t kn = 0; kn < num_pe; kn++) { out1[kn] = out0[kn]; }

            if (output_on) {
              for (uint16_t kn = 0; kn < num_pe; kn++) {
                T_out conv_result = out1[kn];
                T_out out_buf;

                bool last_kernel = (kh == k_h - 1) && (kw == k_w - 1);

                if (threshold_data != NULL && last_kernel) {
                  T_out ts0 = threshold_local[kn][0];
                  T_out ts1 = threshold_local[kn][1];
                  T_out ts2 = threshold_local[kn][2];
                  T_out flag = threshold_local[kn][3];

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
                    out_buf = flag - 2;
                  }
                } else {
                  out_buf = conv_result;
                }

                out_data[idx_out + kn] = out_buf;
              }
              idx_out += out_c;
            }
          }
        }
        delete[] k_local;
      }
    }
  }
} // namespace cpp
} // namespace cpp

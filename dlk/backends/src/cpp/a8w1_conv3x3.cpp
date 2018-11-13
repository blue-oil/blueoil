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

void a8w1_conv3x3_impl(T_in_k2c in_data[], T_out_k2c out_data[], T_k_k2c k_data[], unsigned in_w, unsigned in_h,
                       unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c, unsigned pad, unsigned stride)
{
  namespace p = a8w1_conv3x3_params;

  T_k_k2c k_local[p::k_size * p::k_n];
  int idx_k = 0;

  for (unsigned kn = 0; kn < p::k_n; kn++) {
    for (unsigned k = 0; k < p::k_size; k++) { k_local[k * p::k_n + kn] = k_data[idx_k++]; }
  }

  unsigned idx_out = 0;

  for (unsigned oh = 0; oh < out_h; ++oh)
    for (unsigned ow = 0; ow < out_w; ++ow) {
      unsigned idx_k_local = 0;
      T_out_k2c out[p::k_n];

      for (unsigned i = 0; i < p::k_n; i++) { out[i] = 0; }

      for (unsigned kh = 0; kh < p::k_h; kh++) {
        for (unsigned kw = 0; kw < p::k_w; kw++) {
          int ih = (oh * stride) - pad + kh;
          int iw = (ow * stride) - pad + kw;
          bool valid = (iw >= 0) && (iw < int(in_w)) && (ih >= 0) && (ih < int(in_h));

          for (unsigned kc = 0; kc < p::in_c; kc++) {
            if (valid) {
              int idx_in = ih * in_w * in_c + iw * in_c + kc;
              T_out_k2c in_buf = T_out_k2c(in_data[idx_in]);

              for (int kn = 0; kn < p::k_n; kn++) {
                T_out_k2c k_buf = T_out_k2c(k_local[idx_k_local * p::k_n + kn]);
                out[kn] += in_buf * k_buf;
              }
            }
            idx_k_local++;
          }
        }
      }

      for (int kn = 0; kn < p::k_n; kn++) { out_data[idx_out++] = out[kn]; }
    } // for LOOP_CONV_INPUT
}

void a8w1_conv3x3_with_kn2row_impl(T_in_k2c in_data[], T_out_k2c out_data[], T_k_k2c k_data[],
                                   T_out_k2c out_data_partial[], unsigned in_w, unsigned in_h, unsigned in_c,
                                   unsigned out_w, unsigned out_h, unsigned out_c, unsigned pad)
{
  namespace p = a8w1_conv3x3_params;

  unsigned idx_k = 0;

  for (int8_t kh = 0; kh < p::k_h; kh++) {
    for (int8_t kw = 0; kw < p::k_w; kw++) {
      T_k_k2c k_local[p::in_c * p::out_c];

      for (uint16_t kc = 0; kc < p::in_c; kc++) {
        for (uint16_t kn = 0; kn < p::out_c; kn++) { k_local[kc * p::out_c + kn] = k_data[idx_k++]; }
      }

      unsigned idx_in = 0;
      unsigned idx_out = 0;

      for (int16_t _ih = 0; _ih < in_h + 2 * p::pad_h; _ih++) {
        for (int16_t _iw = 0; _iw < in_w + 2 * p::pad_w; _iw++) {
          int16_t ih = _ih - p::pad_h;
          int16_t iw = _iw - p::pad_w;
          int16_t oh = _ih - kh;
          int16_t ow = _iw - kw;

          bool first_load = ((kh == 0) && (kw == 0));
          bool input_on = ((ih >= 0) && (ih < in_h) && (iw >= 0) && (iw < in_w));
          bool output_on = ((oh >= 0) && (oh < out_h) && (ow >= 0) && (ow < out_w));

          T_out_k2c out[p::out_c];

          for (uint16_t kn = 0; kn < p::out_c; kn++) {
            out[kn] = (output_on && !first_load) ? out_data_partial[idx_out + kn] : 0;
          }

          for (uint16_t ic = 0; ic < p::in_c; ic++) {
            T_out_k2c in = 0;

            if (input_on) {
              in = T_out_k2c(in_data[idx_in++]);
            }

            for (uint16_t kn = 0; kn < p::out_c; kn++) {
              T_out_k2c k = T_out_k2c(k_local[ic * p::out_c + kn]);
              out[kn] += in * k;
            }
          }

          if (output_on) {
            for (uint16_t kn = 0; kn < p::out_c; kn++) { out_data[idx_out + kn] = out[kn]; }
            idx_out += p::out_c;
          }
        }
      }
    }
  }
}
} // namespace cpp

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

#include <cstdio>
#include "intel_hls/config.h"

namespace p = conv_kn2row_params;

T_out_hls PE_kn2row_tiling(uint32 k_buf, uint32 in_buf[2])
{
  T_out_hls out = 0;
  uint6 nk_buf_ppcounut = __builtin_popcount(~k_buf);

#pragma unroll
  for (unsigned ib = 0; ib < p::max_in_b; ib++) {
    // TODO: we can save resources by precomputing the sum of ~k_buf.
    uint6 xnor_popcount = __builtin_popcount(~(in_buf[ib] ^ k_buf));
    out += ((xnor_popcount - nk_buf_ppcounut) << ib);
  }
  return out;
}

hls_avalon_slave_component void intel_hls_qconv_kn2row_tiling_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  ihc::mm_master<T_out_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> > &threshold_data,
  hls_avalon_slave_register_argument int32 in_w, hls_avalon_slave_register_argument int32 in_h,
  hls_avalon_slave_register_argument int32 in_c_by_word, hls_avalon_slave_register_argument int32 out_w,
  hls_avalon_slave_register_argument int32 out_h, hls_avalon_slave_register_argument int32 out_c,
  hls_avalon_slave_register_argument int32 k_w, hls_avalon_slave_register_argument int32 k_h,
  hls_avalon_slave_register_argument int32 pad, hls_avalon_slave_register_argument int32 use_threshold)
{
  /// just alias for better understanding
  static const unsigned out_c_low = p::num_pe;
  assert((out_c % out_c_low) == 0);
  assert(in_c_by_word <= p::max_in_c_by_word);
  assert(in_c_by_word >= p::min_in_c_by_word);
  // assert(in_b <= p::max_in_b);
  // assert(in_b >= p::min_in_b);
  assert(k_h <= p::max_k_h);
  assert(k_h >= p::min_k_h);
  assert(k_w <= p::max_k_w);
  assert(k_w >= p::min_k_w);

  for (int oc_high = 0; oc_high < out_c; oc_high += out_c_low) {
    T_out threshold_buf[out_c_low][p::num_thresholds];

    if (use_threshold > 0) {
      for (unsigned oc = 0; oc < out_c_low; oc++) {
        for (unsigned i = 0; i < p::num_thresholds; i++) {
          unsigned idx_th = (oc_high + oc) * p::num_thresholds + i;
          threshold_buf[oc][i] = threshold_data[idx_th];
        }
      }
    }

    for (int ih_high = 0; ih_high < in_h + 2 * pad; ih_high += p::tile_h) {
      for (int iw_high = 0; iw_high < in_w + 2 * pad; iw_high += p::tile_w) {
        T_in_hls in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c_by_word][p::max_in_b];
        T_out_hls out_buf[p::tile_w][p::tile_w][out_c_low];
        T_k_hls k_buf[p::max_in_c_by_word][out_c_low];

        /// preload input
        for (int ih_low = 0; ih_low < p::in_tile_h; ++ih_low) {
          for (int iw_low = 0; iw_low < p::in_tile_w; ++iw_low) {
            /// index must care the padding, so we skip the padding part that
            /// doesn't exist in actuall memory.
            int ih = (ih_low + ih_high - pad);

            int iw = (iw_low + iw_high - pad);
            bool input_on = (ih >= 0) && (iw >= 0) && (ih < in_h) && (iw < in_w);

            for (int ic = 0; ic < in_c_by_word; ic++) {
              for (int ib = 0; ib < p::max_in_b; ib++) {
                const int _in_w = int(in_w);
                const int _in_c = int(in_c_by_word);
                in_buf[ih_low][iw_low][ic][ib] =
                  (input_on)
                    ? in_data[ih * _in_w * _in_c * p::max_in_b + iw * _in_c * p::max_in_b + ic * p::max_in_b + ib]
                    : T_in_hls(0);
              }
            }
          }
        }

        /// initialize output_buf
        // TODO: this could be done at the same time in the accumuratoin step.
        for (int oh = 0; oh < p::tile_h; ++oh) {
          for (int ow = 0; ow < p::tile_w; ++ow) {
            for (int oc = 0; oc < out_c_low; ++oc) { out_buf[oh][ow][oc] = 0; }
          }
        }

        /// main convolution loop
        for (int kh = 0; kh < k_h; ++kh) {
          for (int kw = 0; kw < k_w; ++kw) {
            /// preload kernel
            for (int ic = 0; ic < in_c_by_word; ic++) {
              for (int ib = 0; ib < p::max_in_b; ib++) {
                for (int oc = 0; oc < out_c_low; oc++) {
                  /// currently kernel oerder is NoHWCNi, which means the
                  /// outermost dimension "N" is split into 2 high and low
                  /// parts. we should be carefull when compute the index.
                  const int _in_c = int(in_c_by_word);
                  const int _out_c = int(out_c);
                  const int _k_w = int(k_w);
                  const int _k_h = int(k_h);
                  int idx_k = (kh * _k_w * _in_c * out_c_low) + (kw * _in_c * out_c_low) + (ic * out_c_low) + oc +
                              (oc_high * _k_h * _k_w * _in_c);
                  k_buf[ic][oc] = k_data[idx_k];
                }
              }
            }

            for (int ih = 0; ih < p::in_tile_h; ++ih) {
              for (int iw = 0; iw < p::in_tile_w; ++iw) {
                int oh = ih - kh;
                int ow = iw - kw;
                bool output_on = (oh >= 0) && (ow >= 0) && (oh < p::tile_h) && (ow < p::tile_w);

                for (int ic = 0; ic < in_c_by_word; ic++) {
                  T_in_hls in_elems[p::max_in_b];
                  for (int ib = 0; ib < p::max_in_b; ib++) { in_elems[ib] = in_buf[ih][iw][ic][ib]; }

                  for (int oc = 0; oc < out_c_low; oc++) {
                    T_k_hls k_elem = k_buf[ic][oc];
                    T_out_hls acc_tmp = PE_kn2row_tiling(k_elem, in_elems);

                    if (output_on) {
                      out_buf[oh][ow][oc] += acc_tmp;
                    }
                  }
                }
              }
            }
          }
        }

        /// export data in output buffer step
        for (int oh = 0; oh < p::tile_h; ++oh) {
          for (int ow = 0; ow < p::tile_w; ++ow) {
            for (int oc = 0; oc < out_c_low; oc++) {
              T_out_hls out = out_buf[oh][ow][oc];
              T_out_hls tmp;

              if (use_threshold > 0) {
                T_out_hls ts0 = threshold_buf[oc][0];
                T_out_hls ts1 = threshold_buf[oc][1];
                T_out_hls ts2 = threshold_buf[oc][2];
                T_out_hls flag = threshold_buf[oc][3];

                if (flag == 1) /// increasing function
                {
                  if (out < ts0)
                    tmp = 0;
                  else if (out < ts1)
                    tmp = 1;
                  else if (out < ts2)
                    tmp = 2;
                  else
                    tmp = 3;
                } else if (flag == -1) /// decreasing function
                {
                  if (out > ts2)
                    tmp = 0;
                  else if (out > ts1)
                    tmp = 1;
                  else if (out > ts0)
                    tmp = 2;
                  else
                    tmp = 3;
                } else {
                  /// max value of 2 bits
                  T_out_hls k = 3 * 3 * out_c * 3;
                  tmp = flag - k;
                }
              } else {
                tmp = out;
              }

              /// export out data to actual memory space.
              unsigned oh_ = ih_high + oh;
              unsigned ow_ = iw_high + ow;
              unsigned oc_ = oc_high + oc;

              bool output_on = ((oh_ < out_h) && (ow_ < out_w) && (oc_ < out_c));
              if (output_on) {
                const int _out_w = int(out_w);
                const int _out_c = int(out_c);
                int idx_out = oh_ * _out_w * _out_c + ow_ * _out_c + oc_;
                // printf("oh: %d, ow: %d, oc: %d, idx: %d, tmp: %d\n", oh_,
                // ow_,
                //        oc_, idx_out, tmp);
                out_data[idx_out] = T_out_hls(tmp);
              }
            }
          }
        }
      }
    }
  }
}
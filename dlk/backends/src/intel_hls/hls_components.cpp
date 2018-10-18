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

#include "intel_hls/config.h"

#define T_buf int16

hls_avalon_slave_component void intel_hls_qconv_with_kn2row_impl(
    hls_avalon_slave_register_argument ihc::mm_master<T_in_hls,
                                                      ihc::aspace<1>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &in_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_out_hls,
                                                      ihc::aspace<2>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &out_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_k_hls,
                                                      ihc::aspace<3>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &k_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_out_hls,
                                                      ihc::aspace<4>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &out_data_partial,
    hls_avalon_slave_register_argument int32 in_w,
    hls_avalon_slave_register_argument int32 in_h,
    hls_avalon_slave_register_argument int32 in_c_by_word,
    hls_avalon_slave_register_argument int32 out_w,
    hls_avalon_slave_register_argument int32 out_h,
    hls_avalon_slave_register_argument int32 out_c,
    hls_avalon_slave_register_argument int32 k_w,
    hls_avalon_slave_register_argument int32 k_h,
    hls_avalon_slave_register_argument int32 pad) {
  static const unsigned num_pe = conv_common_params::num_pe;
  static const unsigned max_in_c_by_word = conv_common_params::max_in_c_by_word;
  static const unsigned nbits_in_data = conv_common_params::nbits_in_data;

  for (char kh = 0; kh < k_h; kh++) {
    for (char kw = 0; kw < k_w; kw++) {
      for (short oc = 0; oc < out_c; oc += num_pe) {
        hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4, 5) T_q k_local[max_in_c_by_word][num_pe];

        for (unsigned short kc = 0; kc < in_c_by_word; kc++) {
#pragma unroll
          for (unsigned short kn = 0; kn < num_pe; kn++) {
            const int _k_w = int(k_w);
            const int _out_c = int(out_c);
            const int _in_c = int(in_c_by_word);
            k_local[kc][kn] = k_data[kh * _k_w * _out_c * _in_c + kw * _out_c * _in_c + oc * _in_c + kc * num_pe + kn];
          }
        }

        for (short _ih = 0; _ih < in_h + 2 * pad; _ih++) {
          for (short _iw = 0; _iw < in_w + 2 * pad; _iw++) {
            const int ih = _ih - pad;
            const int iw = _iw - pad;
            const int oh = _ih - kh;
            const int ow = _iw - kw;

            bool first_load = ((kh == 0) && (kw == 0));
            bool input_on = ((ih >= 0) && (ih < in_h) && (iw >= 0) && (iw < in_w));
            bool output_on = ((oh >= 0) && (oh < out_h) && (ow >= 0) && (ow < out_w));

            // unsigned idx_out = oh * short(out_w) * short(out_c) + ow * short(out_c) + oc;

            hls_register T_buf out0[num_pe];
            hls_register T_buf out1[num_pe];
            hls_register T_buf out2[num_pe];

            if (output_on) {
#pragma unroll
              for (unsigned short kn = 0; kn < num_pe; kn++) {
                const int _out_w = int(out_w);
                const int _out_c = int(out_c);
                T_buf out = out_data_partial[oh * _out_w * _out_c + ow * _out_c + oc + kn];
                out0[kn] = (first_load) ? T_buf(0) : out;
              }
            } else {
#pragma unroll
              for (unsigned short kn = 0; kn < num_pe; kn++) {
                out0[kn] = 0;
              }
            }

#pragma unroll
            for (unsigned short kn = 0; kn < num_pe; kn++) {
              out1[kn] = out0[kn];
            }

#pragma unroll 4
            for (unsigned short ic = 0; ic < in_c_by_word; ic++) {
#pragma unroll
              for (unsigned short ib = 0; ib < nbits_in_data; ib++) {
                T_q in0 = 0;

                if (input_on) {
                  const int _in_c = int(in_c_by_word);
                  const int _in_w = int(in_w);
                  // unsigned idx_in = nbits_in_data * (ih * short(in_w) * in_c + iw * in_c + ic);
                  in0 = in_data[nbits_in_data * (ih * _in_w * _in_c + iw * _in_c + ic) + ib];
                }

                T_q in1 = in0;

#pragma unroll
                for (unsigned short kn = 0; kn < num_pe; kn++) {
                  T_q k = k_local[ic][kn];
                  T_q nk_pop = __builtin_popcount(~k);
                  T_q xnor_pop = __builtin_popcount(~(in1 ^ k));
                  out1[kn] += ((xnor_pop - nk_pop) << ib);
                }
              }
            }

#pragma unroll
            for (unsigned short kn = 0; kn < num_pe; kn++) {
              out2[kn] = out1[kn];
            }

            if (output_on) {
#pragma unroll
              for (unsigned short kn = 0; kn < num_pe; kn++) {
                const int _out_w = int(out_w);
                const int _out_c = int(out_c);
                out_data[oh * _out_w * _out_c + ow * _out_c + oc + kn] = out2[kn];
              }
            }
          }
        }
      }
    }
  }
}

#define T_buf_k2c int16

hls_avalon_slave_component void intel_hls_a8w1_qconv_with_kn2row_impl(
    hls_avalon_slave_register_argument ihc::mm_master<T_in_k2c,
                                                      ihc::aspace<1>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &in_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_out_k2c,
                                                      ihc::aspace<2>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &out_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_k_k2c,
                                                      ihc::aspace<3>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &k_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_out_k2c,
                                                      ihc::aspace<4>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &out_data_partial,
    hls_avalon_slave_register_argument int32 in_w,
    hls_avalon_slave_register_argument int32 in_h,
    hls_avalon_slave_register_argument int32 in_c,
    hls_avalon_slave_register_argument int32 out_w,
    hls_avalon_slave_register_argument int32 out_h,
    hls_avalon_slave_register_argument int32 out_c,
    hls_avalon_slave_register_argument int32 kw,
    hls_avalon_slave_register_argument int32 kh,
    hls_avalon_slave_register_argument int32 pad) {
  namespace p = a8w1_conv3x3_params;

  hls_register T_k_k2c k_local[p::in_c][p::num_pe];

  for (unsigned short kc = 0; kc < p::in_c; kc++) {
#pragma unroll
    for (unsigned short kn = 0; kn < p::num_pe; kn++) {
      k_local[kc][kn] = k_data[kc * p::num_pe + kn];
    }
  }

  for (short _ih = 0; _ih < in_h + 2 * p::pad_h; _ih++) {
    for (short _iw = 0; _iw < in_w + 2 * p::pad_w; _iw++) {
      const int ih = _ih - p::pad_h;
      const int iw = _iw - p::pad_w;
      const int oh = _ih - short(kh);
      const int ow = _iw - short(kw);

      bool first_load = ((kh == 0) && (kw == 0));
      bool input_on = ((ih >= 0) && (ih < in_h) && (iw >= 0) && (iw < in_w));
      bool output_on = ((oh >= 0) && (oh < out_h) && (ow >= 0) && (ow < out_w));

      hls_register T_buf_k2c out[p::out_c] = {0, 0, 0, 0, 0, 0, 0, 0};

      unsigned idx_in = ih * int(in_w) * p::in_c + iw * p::in_c;
      unsigned idx_k = 0;

      T_out_k2c in0 = 0;
      T_out_k2c in1 = 0;
      T_out_k2c in2 = 0;

      if (input_on) {
        in0 = T_out_k2c(in_data[idx_in + 0]);
        in1 = T_out_k2c(in_data[idx_in + 1]);
        in2 = T_out_k2c(in_data[idx_in + 2]);
      }

#pragma unroll
      for (unsigned short kn = 0; kn < p::num_pe; kn++) {
        T_out_k2c k0 = T_out_k2c(k_local[idx_k + 0][kn]);
        T_out_k2c k1 = T_out_k2c(k_local[idx_k + 1][kn]);
        T_out_k2c k2 = T_out_k2c(k_local[idx_k + 2][kn]);

        out[kn] += in0 * k0 + in1 * k1 + in2 * k2;
      }

      if (output_on) {
        unsigned idx_out = oh * int(out_w) * p::out_c + ow * p::out_c;

        T_buf_k2c out_pre0 = out_data_partial[idx_out + 0];
        T_buf_k2c out_pre1 = out_data_partial[idx_out + 1];
        T_buf_k2c out_pre2 = out_data_partial[idx_out + 2];
        T_buf_k2c out_pre3 = out_data_partial[idx_out + 3];
        T_buf_k2c out_pre4 = out_data_partial[idx_out + 4];
        T_buf_k2c out_pre5 = out_data_partial[idx_out + 5];
        T_buf_k2c out_pre6 = out_data_partial[idx_out + 6];
        T_buf_k2c out_pre7 = out_data_partial[idx_out + 7];

        if (first_load) {
          out_pre0 = (first_load) ? T_buf_k2c(0) : out_pre0;
          out_pre1 = (first_load) ? T_buf_k2c(0) : out_pre1;
          out_pre2 = (first_load) ? T_buf_k2c(0) : out_pre2;
          out_pre3 = (first_load) ? T_buf_k2c(0) : out_pre3;
          out_pre4 = (first_load) ? T_buf_k2c(0) : out_pre4;
          out_pre5 = (first_load) ? T_buf_k2c(0) : out_pre5;
          out_pre6 = (first_load) ? T_buf_k2c(0) : out_pre6;
          out_pre7 = (first_load) ? T_buf_k2c(0) : out_pre7;
        }

        out_data[idx_out + 0] = out_pre0 + out[0];
        out_data[idx_out + 1] = out_pre1 + out[1];
        out_data[idx_out + 2] = out_pre2 + out[2];
        out_data[idx_out + 3] = out_pre3 + out[3];
        out_data[idx_out + 4] = out_pre4 + out[4];
        out_data[idx_out + 5] = out_pre5 + out[5];
        out_data[idx_out + 6] = out_pre6 + out[6];
        out_data[idx_out + 7] = out_pre7 + out[7];
      }
    }
  }
}

hls_avalon_slave_component void intel_hls_apply_thresholds_impl(
    hls_avalon_slave_register_argument ihc::mm_master<T_out_hls,
                                                      ihc::aspace<1>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &in_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_out_hls,
                                                      ihc::aspace<2>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &out_data,
    hls_avalon_slave_register_argument ihc::mm_master<T_out_hls,
                                                      ihc::aspace<3>,
                                                      ihc::awidth<32>,
                                                      ihc::dwidth<128>,
                                                      ihc::latency<0>,
                                                      ihc::maxburst<32>,
                                                      ihc::align<16>,
                                                      ihc::waitrequest<true> > &th_data,
    hls_avalon_slave_register_argument int32 out_w,
    hls_avalon_slave_register_argument int32 out_h,
    hls_avalon_slave_register_argument int32 out_c) {

  static const unsigned num_th = conv_common_params::num_thresholds;
  static const unsigned max_in_c = conv_common_params::max_in_c;
  hls_memory hls_singlepump hls_bankbits(0,1,2,3,4) T_out_hls th_local[max_in_c][num_th];

  for (short oc = 0; oc < out_c; oc++) {
#pragma unroll
    for (char i = 0; i < num_th; i++) {
      th_local[oc][i] = th_data[oc * num_th + i];
    }
  }

  unsigned idx_out = 0;

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
#pragma unroll 8
      for (int oc = 0; oc < out_c; oc++) {
        int out_w_ = int(out_w);
        int out_c_ = int(out_c);
        int idx_out = oh * out_w_ * out_c_ + ow * out_c_ + oc;
        T_buf x = in_data[idx_out];
        T_buf out_buf;

        T_buf t0 = th_local[oc][0];
        T_buf t1 = th_local[oc][1];
        T_buf t2 = th_local[oc][2];
        T_buf flag = th_local[oc][3];

        if (flag == 1)  // increasing function
        {
          if (x < t0)
            out_buf = 0;
          else if (x < t1)
            out_buf = 1;
          else if (x < t2)
            out_buf = 2;
          else
            out_buf = 3;
        } else if (flag == -1)  // decreasing function
        {
          if (x > t2)
            out_buf = 0;
          else if (x > t1)
            out_buf = 1;
          else if (x > t0)
            out_buf = 2;
          else
            out_buf = 3;
        } else {
          // max value of 2 bit
          out_buf = flag - 2;                 // note: 2 is a magic number!
        }

        out_data[idx_out] = out_buf;
      }
    }
  }
}

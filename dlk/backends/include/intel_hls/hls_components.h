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

#pragma once
#include <cassert>
#include "intel_hls/config.h"

namespace p = conv3x3_params;

hls_avalon_slave_component void intel_hls_qconv1x1_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument uint32 in_w, hls_avalon_slave_register_argument uint32 in_h,
  hls_avalon_slave_register_argument uint32 in_c, hls_avalon_slave_register_argument uint32 out_w,
  hls_avalon_slave_register_argument uint32 out_h, hls_avalon_slave_register_argument uint32 out_c,
  hls_avalon_slave_register_argument uint32 out_c_offset);

hls_avalon_slave_component void intel_hls_qconv3x3_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument uint32 in_w, hls_avalon_slave_register_argument uint32 in_h,
  hls_avalon_slave_register_argument uint32 in_c_by_word, hls_avalon_slave_register_argument uint32 out_w,
  hls_avalon_slave_register_argument uint32 out_h, hls_avalon_slave_register_argument uint32 out_c,
  hls_avalon_slave_register_argument uint32 out_c_offset);

void intel_hls_qconv(unsigned k_w, unsigned k_h, T_q in_data_packed[], T_out out_data[], T_q k_data_packed[],
                     unsigned in_w, unsigned in_h, unsigned in_c_by_word, unsigned nbits_in_data, unsigned out_w,
                     unsigned out_h, unsigned out_c, unsigned pad, unsigned stride)
{
  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;

  ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>,
                 ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> >
    avmm_in(in_data_packed, (in_size + in_w * in_c_by_word) * sizeof(T_in_hls));

  ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                 ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> >
    avmm_out(out_data, out_size * sizeof(T_out_hls));

  ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<32>, ihc::waitrequest<true> >
    avmm_k(k_data_packed, k_size * sizeof(T_k_hls));

  if (k_h == 3 && k_w == 3) {
    intel_hls_qconv3x3_impl(avmm_in, avmm_out, avmm_k, in_w, in_h, in_c_by_word, out_w, out_h, out_c, 0);
  } else if (k_h == 1 && k_w == 1) {
    intel_hls_qconv1x1_impl(avmm_in, avmm_out, avmm_k, in_w, in_h, in_c_by_word, out_w, out_h, out_c, 0);
  } else {
    std::cout << "conv" << k_h << "x" << k_w << "is not supported..." << std::endl;
  }
}

hls_avalon_slave_component void intel_hls_qgemm_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_A_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &A_packed,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_B_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &B_packed,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_Y_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> > &Y,
  hls_avalon_slave_register_argument uint32 a_row, hls_avalon_slave_register_argument uint32 a_col_by_word,
  hls_avalon_slave_register_argument uint32 b_col);

void intel_hls_qgemm(T_q A_packed[], T_q B_packed[], T_out Y[], type::u32 a_row, type::u32 a_col_by_word,
                     type::u32 b_col, type::u32 nbits_a, type::u32 nbits_b)
{
  assert(nbits_a == 2);
  assert(nbits_b == 1);

  const unsigned b_row = a_col_by_word;
  const unsigned y_row = a_row;
  const unsigned y_col = b_col;

  const unsigned a_size = a_row * a_col_by_word * nbits_a;
  const unsigned b_size = b_row * b_col * nbits_b;
  const unsigned y_size = y_row * y_col;

  ihc::mm_master<T_A_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<32>, ihc::waitrequest<true> >
    avmm_a(A_packed, a_size * sizeof(T_A_hls));

  ihc::mm_master<T_B_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<32>, ihc::waitrequest<true> >
    avmm_b(B_packed, b_size * sizeof(T_B_hls));

  ihc::mm_master<T_Y_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                 ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true> >
    avmm_y(Y, y_size * sizeof(T_Y_hls));

  intel_hls_qgemm_impl(avmm_a, avmm_b, avmm_y, a_row, a_col_by_word, b_col);
}

hls_avalon_slave_component void intel_hls_qconv_with_kn2row_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data_partial,
  hls_avalon_slave_register_argument int32 in_w, hls_avalon_slave_register_argument int32 in_h,
  hls_avalon_slave_register_argument int32 in_c_by_word, hls_avalon_slave_register_argument int32 out_w,
  hls_avalon_slave_register_argument int32 out_h, hls_avalon_slave_register_argument int32 out_c,
  hls_avalon_slave_register_argument int32 k_w, hls_avalon_slave_register_argument int32 k_h,
  hls_avalon_slave_register_argument int32 pad);

hls_avalon_slave_component void intel_hls_apply_thresholds_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &th_data,
  hls_avalon_slave_register_argument int32 out_w, hls_avalon_slave_register_argument int32 out_h,
  hls_avalon_slave_register_argument int32 out_c);

void intel_hls_qconv_with_kn2row(T_q in_data_packed[], T_out out_data[], T_q k_data_packed[], T_out th_data[],
                                 unsigned in_w, unsigned in_h, unsigned in_c_by_word, unsigned nbits_in_data,
                                 unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w, unsigned k_h,
                                 unsigned pad, unsigned stride)
{
  assert(((k_h == 3) && (k_w == 3)) || ((k_h == 1) && (k_w == 1)));
  assert(((k_h == 3) && (pad == 1)) || ((k_h == 1) && (pad == 0)));
  assert(stride == 1);

  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;

  ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_in(in_data_packed, (in_size + in_w * in_c_by_word) * sizeof(T_in_hls));

  ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_k(k_data_packed, k_size * sizeof(T_k_hls));

  if (th_data != NULL) {
    T_out *out_buf = new T_out[out_size];

    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_out(out_buf, out_size * sizeof(T_out_hls));

    ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_out_partial(out_buf, out_size * sizeof(T_out_hls));

    intel_hls_qconv_with_kn2row_impl(avmm_in, avmm_out, avmm_k, avmm_out_partial, in_w, in_h, in_c_by_word, out_w,
                                     out_h, out_c, k_h, k_w, pad);

    ihc::mm_master<T_out_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_out_after_conv(out_buf, out_size * sizeof(T_out_hls));

    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_out_last(out_data, out_size * sizeof(T_out_hls));

    const unsigned num_th = conv_common_params::num_thresholds;
    ihc::mm_master<T_out_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_th(th_data, out_c * num_th * sizeof(T_out_hls));

    intel_hls_apply_thresholds_impl(avmm_out_after_conv, avmm_out_last, avmm_th, out_w, out_h, out_c);
  } else {
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_out(out_data, out_size * sizeof(T_out_hls));

    ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> >
      avmm_out_partial(out_data, out_size * sizeof(T_out_hls));

    intel_hls_qconv_with_kn2row_impl(avmm_in, avmm_out, avmm_k, avmm_out_partial, in_w, in_h, in_c_by_word, out_w,
                                     out_h, out_c, k_h, k_w, pad);
  }
}

void intel_hls_a8w1_qconv_with_kn2row_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_k2c, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_k2c, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_k2c, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_k2c, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data_partial,
  hls_avalon_slave_register_argument int32 in_w, hls_avalon_slave_register_argument int32 in_h,
  hls_avalon_slave_register_argument int32 in_c, hls_avalon_slave_register_argument int32 out_w,
  hls_avalon_slave_register_argument int32 out_h, hls_avalon_slave_register_argument int32 out_c,
  hls_avalon_slave_register_argument int32 k_w, hls_avalon_slave_register_argument int32 k_h,
  hls_avalon_slave_register_argument int32 pad);

void intel_hls_a8w1_qconv_with_kn2row(T_in_k2c in_data[], T_out_k2c out_data[], T_k_k2c k_data[], unsigned in_w,
                                      unsigned in_h, unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c,
                                      unsigned k_w, unsigned k_h, unsigned pad, unsigned stride)
{
  assert((k_h == 3) && (k_w == 3));
  assert(((k_h == 3) && (pad == 1)) || ((k_h == 1) && (pad == 0)));
  assert(stride == 1);

  const unsigned in_size = in_h * in_w * in_c;
  const unsigned out_size = out_h * out_w * out_c;

  T_out_k2c *out0 = new T_out_k2c[out_size];
  T_out_k2c *out1 = new T_out_k2c[out_size];

  ihc::mm_master<T_in_k2c, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_in(in_data, (in_size + in_w * in_c) * sizeof(T_in_k2c));

  for (unsigned char kh = 0; kh < k_h; kh++) {
    for (unsigned char kw = 0; kw < k_w; kw++) {
      const unsigned k_size = in_c * out_c;
      const unsigned k_offset = kh * k_w * k_size + kw * k_size;
      ihc::mm_master<T_k_k2c, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                     ihc::align<16>, ihc::waitrequest<true> >
        avmm_k(&k_data[k_offset], k_size * sizeof(T_k_k2c));

      if ((kh * k_w + kw) % 2 == 0) {
        ihc::mm_master<T_out_k2c, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> >
          avmm_out(out0, out_size * sizeof(T_out_k2c));

        ihc::mm_master<T_out_k2c, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> >
          avmm_out_partial(out1, out_size * sizeof(T_out_k2c));

        intel_hls_a8w1_qconv_with_kn2row_impl(avmm_in, avmm_out, avmm_k, avmm_out_partial, in_w, in_h, in_c, out_w,
                                              out_h, out_c, kw, kh, pad);
      } else {
        ihc::mm_master<T_out_k2c, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> >
          avmm_out(out1, out_size * sizeof(T_out_k2c));

        ihc::mm_master<T_out_k2c, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> >
          avmm_out_partial(out0, out_size * sizeof(T_out_k2c));

        intel_hls_a8w1_qconv_with_kn2row_impl(avmm_in, avmm_out, avmm_k, avmm_out_partial, in_w, in_h, in_c, out_w,
                                              out_h, out_c, kw, kh, pad);
      }
    }
  }

  if ((k_h * k_w) % 2 == 0) {
    for (unsigned i = 0; i < out_size; i++) { out_data[i] = out1[i]; }
  } else {
    for (unsigned i = 0; i < out_size; i++) { out_data[i] = out0[i]; }
  }
}

hls_avalon_slave_component void intel_hls_qconv_kn2row_tiling_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &k_data,
  ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> > &threshold_data,
  hls_avalon_slave_register_argument int32 in_w, hls_avalon_slave_register_argument int32 in_h,
  hls_avalon_slave_register_argument int32 in_c_by_word, hls_avalon_slave_register_argument int32 out_w,
  hls_avalon_slave_register_argument int32 out_h, hls_avalon_slave_register_argument int32 out_c,
  hls_avalon_slave_register_argument int32 k_w, hls_avalon_slave_register_argument int32 k_h,
  hls_avalon_slave_register_argument int32 pad, hls_avalon_slave_register_argument int32 use_threshold);

void intel_hls_qconv_kn2row_tiling(T_q in_data_packed[], T_out out_data[], T_q k_data_packed[], T_out th_data[],
                                   unsigned in_w, unsigned in_h, unsigned in_c_by_word, unsigned nbits_in_data,
                                   unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w, unsigned k_h,
                                   unsigned pad, unsigned stride)
{
  assert(((k_h == 3) && (k_w == 3)) || ((k_h == 1) && (k_w == 1)));
  assert(((k_h == 3) && (pad == 1)) || ((k_h == 1) && (pad == 0)));
  assert(stride == 1);

  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;
  const unsigned num_th = conv_common_params::num_thresholds;
  const int use_threshold = (th_data != NULL) ? 1 : 0;

  ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_in(in_data_packed, (in_size + in_w * in_c_by_word) * sizeof(T_in_hls));

  ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_out(out_data, out_size * sizeof(T_out_hls));

  ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_k(k_data_packed, k_size * sizeof(T_k_hls));

  ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                 ihc::align<16>, ihc::waitrequest<true> >
    avmm_th(th_data, out_c * num_th * sizeof(T_out_hls));

  intel_hls_qconv_kn2row_tiling_impl(avmm_in, avmm_out, avmm_k, avmm_th, in_w, in_h, in_c_by_word, out_w, out_h, out_c,
                                     k_h, k_w, pad, use_threshold);
}
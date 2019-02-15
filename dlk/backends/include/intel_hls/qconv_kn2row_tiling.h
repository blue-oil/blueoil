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
  hls_avalon_slave_register_argument
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
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

namespace p = conv1x1_params;

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
  hls_avalon_slave_register_argument uint32 out_c_offset)
{
  unsigned idx_k = 0;

  for (unsigned oc = 0; oc < out_c; oc += p::num_pe) {
    hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4, 5, 6) T_k_hls k_local[p::max_in_c][p::num_pe];

#pragma unroll 4
    for (unsigned kc = 0; kc < in_c; kc++) {
#pragma unroll
      for (unsigned kn = 0; kn < p::num_pe; kn++) { k_local[kc][kn] = k_data[idx_k++]; }
    }

    unsigned idx_out = oc;
    unsigned idx_in = 0;

#pragma loop_coalesce 2
    for (unsigned ih = 0; ih < in_h; ih++) {
      for (unsigned iw = 0; iw < in_w; iw++) {
#pragma unroll 8
        hls_register int16 out0[p::num_pe] = {0, 0, 0, 0, 0, 0, 0, 0};

        hls_register int16 out1[p::num_pe] = {0, 0, 0, 0, 0, 0, 0, 0};

        for (unsigned ic = 0; ic < in_c; ic++) {
#pragma unroll
          for (unsigned ib = 0; ib < p::nbits_in_data; ib++) {
            T_in_hls in = in_data[idx_in++];

#pragma unroll
            for (unsigned np = 0; np < p::num_pe; np++) {
              T_k_hls k = k_local[ic][np];
              int8 nk_pop = __builtin_popcount(~k);
              int8 xnor_pop = __builtin_popcount(~(in ^ k));
              out0[np] += ((xnor_pop - nk_pop) << ib);
            }
          }
        }
#pragma unroll
        for (unsigned np = 0; np < p::num_pe; np++) { out1[np] = out0[np]; }
#pragma unroll
        for (unsigned np = 0; np < p::num_pe; np++) { out_data[idx_out + np] = out1[np]; }
        idx_out += out_c;
      }
    }
  }
}

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

namespace p = conv3x3_params;

T_out_hls PE(uint32 k_buf, uint32 in_buf[2])
{
  T_out_hls out = 0;
  uint6 nk_buf_ppcounut = __builtin_popcount(~k_buf);

#pragma unroll
  for (unsigned ib = 0; ib < p::nbits_in_data; ib++) {
    uint6 xnor_popcount = __builtin_popcount(~(in_buf[ib] ^ k_buf));
    out += ((xnor_popcount - nk_buf_ppcounut) << ib);
  }
  return out;
}

hls_avalon_slave_component void intel_hls_qconv3x3_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true>> &in_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true>> &out_data,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true>> &k_data,
  hls_avalon_slave_register_argument uint32 in_w, hls_avalon_slave_register_argument uint32 in_h,
  hls_avalon_slave_register_argument uint32 in_c_by_word, hls_avalon_slave_register_argument uint32 out_w,
  hls_avalon_slave_register_argument uint32 out_h, hls_avalon_slave_register_argument uint32 out_c,
  hls_avalon_slave_register_argument uint32 out_c_offset)
{
  unsigned idx_k = 0;

  for (int oc = 0; oc < out_c; oc += p::num_pe) {
    hls_memory hls_singlepump hls_bankbits(0, 1, 2)
      T_in_hls in_mem_0[p::max_in_w_with_pad][p::max_in_c_by_word][p::nbits_in_data];

    hls_memory hls_singlepump hls_bankbits(0, 1, 2)
      T_in_hls in_mem_1[p::max_in_w_with_pad][p::max_in_c_by_word][p::nbits_in_data];

    hls_memory hls_singlepump hls_bankbits(0, 1, 2)
      T_in_hls in_mem_2[p::max_in_w_with_pad][p::max_in_c_by_word][p::nbits_in_data];

    hls_memory hls_singlepump hls_bankbits(0, 1, 2)
      T_in_hls in_mem_3[p::max_in_w_with_pad][p::max_in_c_by_word][p::nbits_in_data];

    hls_register T_in_hls in_reg[p::k_h][p::k_w][p::max_in_c_by_word][p::nbits_in_data];

    hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4, 5)
      T_k_hls k_local[p::inb_h][p::inb_w][p::max_in_c_by_word][p::num_pe];

    unsigned idx_out = oc;
    unsigned idx_in = 0;

#pragma unroll 1
    for (unsigned kh = 0; kh < p::k_h; kh++) {
#pragma unroll 1
      for (unsigned kw = 0; kw < p::k_w; kw++) {
#pragma unroll 8
        for (unsigned kc = 0; kc < in_c_by_word; kc++) {
#pragma unroll
          for (unsigned kn = 0; kn < p::num_pe; kn++) { k_local[kh][kw][kc][kn] = (T_in_hls)k_data[idx_k++]; }
        }
      }
    }

    for (unsigned ih = 0; ih < in_h + 2 * p::pad_h + 1; ih++)
      for (unsigned iw = 0; iw < in_w + 2 * p::pad_w; iw++) {
        int oh = ih - p::k_h; // -1
        int ow = iw - p::k_w + 1;
        bool input_on = ((ih >= p::pad_h) && (ih < in_h + p::pad_h) && (iw >= p::pad_w) && (iw < in_w + p::pad_w));
        bool output_on = ((oh >= 0) && (ow >= 0));

        hls_register int16 out[p::num_pe] = {0};

        for (int ic = 0; ic < in_c_by_word; ic++) {
#pragma unroll
          for (int ib = 0; ib < p::nbits_in_data; ib++) {
            T_in_hls data = in_data[idx_in + ic * p::nbits_in_data + ib];

            switch (ih % p::inb_h) {
              case 0:
                in_mem_0[iw][ic][ib] = (input_on) ? data : T_in_hls(0);
                break;
              case 1:
                in_mem_1[iw][ic][ib] = (input_on) ? data : T_in_hls(0);
                break;
              case 2:
                in_mem_2[iw][ic][ib] = (input_on) ? data : T_in_hls(0);
                break;
              case 3:
                in_mem_3[iw][ic][ib] = (input_on) ? data : T_in_hls(0);
                break;
            }

            in_reg[0][0][ic][ib] = in_reg[0][1][ic][ib];
            in_reg[1][0][ic][ib] = in_reg[1][1][ic][ib];
            in_reg[2][0][ic][ib] = in_reg[2][1][ic][ib];
            in_reg[0][1][ic][ib] = in_reg[0][2][ic][ib];
            in_reg[1][1][ic][ib] = in_reg[1][2][ic][ib];
            in_reg[2][1][ic][ib] = in_reg[2][2][ic][ib];

            switch (oh % p::inb_h) {
              case 0:
                in_reg[0][2][ic][ib] = in_mem_0[iw][ic][ib];
                in_reg[1][2][ic][ib] = in_mem_1[iw][ic][ib];
                in_reg[2][2][ic][ib] = in_mem_2[iw][ic][ib];
                break;
              case 1:
                in_reg[0][2][ic][ib] = in_mem_1[iw][ic][ib];
                in_reg[1][2][ic][ib] = in_mem_2[iw][ic][ib];
                in_reg[2][2][ic][ib] = in_mem_3[iw][ic][ib];
                break;
              case 2:
                in_reg[0][2][ic][ib] = in_mem_2[iw][ic][ib];
                in_reg[1][2][ic][ib] = in_mem_3[iw][ic][ib];
                in_reg[2][2][ic][ib] = in_mem_0[iw][ic][ib];
                break;
              case 3:
                in_reg[0][2][ic][ib] = in_mem_3[iw][ic][ib];
                in_reg[1][2][ic][ib] = in_mem_0[iw][ic][ib];
                in_reg[2][2][ic][ib] = in_mem_1[iw][ic][ib];
                break;
            }

#pragma unroll
            for (unsigned kh = 0; kh < p::k_h; kh++) {
#pragma unroll
              for (unsigned kw = 0; kw < p::k_w; kw++) {
#pragma unroll
                for (unsigned kn = 0; kn < p::num_pe; kn++) {
                  T_in_hls in = in_reg[kh][kw][ic][ib];
                  T_k_hls k = k_local[kh][kw][ic][kn];
                  uint6 nk_pop = __builtin_popcount(~k);
                  uint6 xnor_pop = __builtin_popcount(~(in ^ k));
                  out[kn] += (xnor_pop - nk_pop) << ib;
                }
              }
            }
          }
        }

        if (input_on) {
          idx_in += in_c_by_word * p::nbits_in_data;
        }

#pragma unroll
        for (int oc = 0; oc < p::num_pe; oc++) { out_data[idx_out + oc] = out[oc]; }

        if (output_on) {
          idx_out += out_c;
        }
      }
  }
}

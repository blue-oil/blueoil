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
#include "HLS/extendedmath.h"
#include "intel_hls/config.h"

namespace p = conv_kn2row_params;

/// just alias for better understanding
static const unsigned out_c_low = p::num_pe;

void input_load_module(ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>,
                                      ihc::maxburst<32>, ihc::align<16>, ihc::waitrequest<true> > &in_data,
                       T_in_hls in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c_by_word][p::max_in_b],
                       const unsigned in_h, const unsigned in_w, const unsigned in_c, const unsigned ih_high,
                       const unsigned iw_high, const unsigned pad)
{
#pragma loop_coalesce 4
#pragma unroll 1
  for (unsigned ih_low = 0; ih_low < p::in_tile_h; ++ih_low) {
#pragma unroll 1
    for (unsigned iw_low = 0; iw_low < p::in_tile_w; ++iw_low) {
#pragma unroll 2
      for (unsigned ic = 0; ic < in_c; ic++) {
#pragma unroll
        for (unsigned ib = 0; ib < p::max_in_b; ib++) {
          /// index must care the padding, so we skip the padding part that
          /// doesn't exist in actual memory.
          const int ih = int(ih_low + ih_high - pad);
          const int iw = int(iw_low + iw_high - pad);
          const bool input_on = (ih >= 0) && (iw >= 0) && (ih < in_h) && (iw < in_w);

          // loading inputs from bus.
          // if the coordinates on the padding, this stores 0 instead of loading the data.
          in_buf[ih_low][iw_low][ic][ib] =
            (input_on) ? in_data[ih * in_w * in_c * p::max_in_b + iw * in_c * p::max_in_b + ic * p::max_in_b + ib]
                       : T_in_hls(0);
        }
      }
    }
  }
}

void kernel_load_module(ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>,
                                       ihc::maxburst<32>, ihc::align<16>, ihc::waitrequest<true> > &k_data,
                        T_k_hls k_buf[p::max_k_h][p::max_k_w][p::max_in_c_by_word][out_c_low], const unsigned k_h,
                        const unsigned k_w, const unsigned k_c, const unsigned kn_high)
{
#pragma loop_coalesce 4
#pragma unroll 1
  for (unsigned kh = 0; kh < k_h; ++kh) {
#pragma unroll 1
    for (unsigned kw = 0; kw < k_w; ++kw) {
#pragma unroll 1
      for (unsigned kc = 0; kc < k_c; kc++) {
#pragma unroll 4
        for (unsigned kn = 0; kn < out_c_low; kn++) {
          /// currently kernel order is NoHWCNi, which means the
          /// outermost dimension "N" is split 2 high and low
          /// parts. we should be carefull when compute the index.
          const unsigned idx_k =
            (kh * k_w * k_c * out_c_low) + (kw * k_c * out_c_low) + (kc * out_c_low) + kn + (kn_high * k_h * k_w * k_c);
          k_buf[kh][kw][kc][kn] = k_data[idx_k];
        }
      }
    }
  }
}

void threshold_load_module(ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>,
                                          ihc::maxburst<32>, ihc::align<16>, ihc::waitrequest<true> > &th_data,
                           T_out_hls th_buf[out_c_low][p::num_thresholds], const unsigned oc_high)
{
#pragma loop_coalesce 2
#pragma unroll 2
  for (unsigned oc = 0; oc < out_c_low; oc++) {
#pragma unroll
    for (unsigned i = 0; i < p::num_thresholds; i++) {
      unsigned idx_th = (oc_high + oc) * p::num_thresholds + i;
      th_buf[oc][i] = th_data[idx_th];
    }
  }
}

void mac_compute_module(T_in_hls in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c_by_word][p::max_in_b],
                        T_out_hls out_buf[p::tile_w][p::tile_w][out_c_low],
                        T_k_hls k_buf[p::max_k_h][p::max_k_w][p::max_in_c_by_word][out_c_low],
                        T_out_hls threshold_buf[out_c_low][p::num_thresholds], const unsigned in_c, const unsigned k_h,
                        const unsigned k_w)
{
#pragma loop_coalesce 2
#pragma unroll 1
  for (unsigned kh = 0; kh < k_h; ++kh) {
#pragma unroll 1
    for (unsigned kw = 0; kw < k_w; ++kw) {

#pragma loop_coalesce 2
#pragma unroll 1
      for (unsigned oh = 0; oh < p::tile_h; ++oh) {
#pragma unroll 1
        for (unsigned ow = 0; ow < p::tile_w; ++ow) {
          hls_register T_out_hls out_regs[out_c_low];

#pragma unroll
          for (unsigned oc = 0; oc < out_c_low; oc++) {
            const bool out_init = ((kh == 0) && (kw == 0));
            const T_out_hls out = out_buf[oh][ow][oc];
            out_regs[oc] = (out_init) ? T_out_hls(0) : out;
          }

#pragma loop_coalesce 2
#pragma unroll 4
          for (unsigned ic = 0; ic < in_c; ic++) {
            const unsigned ih = oh + kh;
            const unsigned iw = ow + kw;
            const T_in_hls in_reg0 = in_buf[ih][iw][ic][0];
            const T_in_hls in_reg1 = in_buf[ih][iw][ic][1];

#pragma unroll
            for (unsigned oc = 0; oc < out_c_low; oc++) {
              const T_k_hls k_reg = k_buf[kh][kw][ic][oc];
              T_out_hls not_k = popcountll(~k_reg);
              T_out_hls xnor0 = popcountll(~(in_reg0 ^ k_reg)) - not_k;
              T_out_hls xnor1 = (popcountll(~(in_reg1 ^ k_reg)) - not_k) << 1;
              out_regs[oc] += (xnor0 + xnor1);
            }
          }

#pragma unroll
          for (unsigned oc = 0; oc < out_c_low; oc++) { out_buf[oh][ow][oc] = out_regs[oc]; }
        }
      }
    }
  }
}

void output_store_module(hls_avalon_slave_register_argument
                           ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>,
                                          ihc::maxburst<32>, ihc::align<16>, ihc::waitrequest<true> > &out_data,
                         T_out_hls out_buf[p::tile_w][p::tile_w][out_c_low],
                         T_out_hls th_buf[out_c_low][p::num_thresholds], const unsigned out_h, const unsigned out_w,
                         const unsigned out_c, const unsigned ih_high, const unsigned iw_high, const unsigned oc_high,
                         const bool use_threshold)
{
#pragma loop_coalesce 3
#pragma unroll 1
  for (unsigned oh = 0; oh < p::tile_h; ++oh) {
#pragma unroll 1
    for (unsigned ow = 0; ow < p::tile_w; ++ow) {
#pragma unroll 8
      for (unsigned oc = 0; oc < out_c_low; oc++) {
        const T_out_hls out = out_buf[oh][ow][oc];
        T_out_hls tmp;

      THRESHOLD_APPLY_MODULE:
        if (use_threshold > 0) {
          const T_out_hls ts0 = th_buf[oc][0];
          const T_out_hls ts1 = th_buf[oc][1];
          const T_out_hls ts2 = th_buf[oc][2];
          const T_out_hls flag = th_buf[oc][3];

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
            const T_out_hls k = 3 * 3 * out_c * 3;
            tmp = flag - k;
          }
        } else {
          tmp = out;
        }

        /// export out data to actual memory space.
        const unsigned oh_ = ih_high + oh;
        const unsigned ow_ = iw_high + ow;
        const unsigned oc_ = oc_high + oc;

        const bool output_on = ((oh_ < out_h) && (ow_ < out_w) && (oc_ < out_c));
        if (output_on) {
          const int idx_out = oh_ * out_w * out_c + ow_ * out_c + oc_;
          out_data[idx_out] = T_out_hls(tmp);
        }
      }
    }
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
  hls_avalon_slave_register_argument
    ihc::mm_master<T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<BW_>, ihc::latency<0>, ihc::maxburst<32>,
                   ihc::align<16>, ihc::waitrequest<true> > &threshold_data,
  hls_avalon_slave_register_argument int32 in_w, hls_avalon_slave_register_argument int32 in_h,
  hls_avalon_slave_register_argument int32 in_c_by_word, hls_avalon_slave_register_argument int32 out_w,
  hls_avalon_slave_register_argument int32 out_h, hls_avalon_slave_register_argument int32 out_c,
  hls_avalon_slave_register_argument int32 k_w, hls_avalon_slave_register_argument int32 k_h,
  hls_avalon_slave_register_argument int32 pad, hls_avalon_slave_register_argument int32 use_threshold)
{
  // in_buf shoule be banked by 8 elems, because this has 2 bits per an element, and
  // 4 inputs are computed along with input channel dimension at a cycle.
  // This also should be doublepump because this loads next data from bus while computing the others.
  hls_memory hls_singlepump hls_bankbits(0, 1, 2)
    T_in_hls in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c_by_word][p::max_in_b];

  // k_buf shoule be banked by 64 elems, because
  // 16 kernels are needed to produce 16 outputs which is fully banked by 16 on out_c_low
  // Also per 1 output, 4 kernels are additionally needed to product with the 8 inputs coming from bus.
  // Only singlepump is OK for kernel.
  hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4, 5)
    hls_memory T_k_hls k_buf[p::max_k_h][p::max_k_w][p::max_in_c_by_word][out_c_low];

  // out_buf shoule be banked by 16 elems, because out_c_low is 16, which log2 is 4.
  // This also should should be doublepump, because accumulation happens at every cycle,
  // requiring reading a data and computing it, then rewriting it to the same address.
  hls_memory hls_doublepump hls_bankbits(0, 1, 2, 3) T_out_hls out_buf[p::tile_w][p::tile_w][out_c_low];

#pragma loop_coalesce 2
#pragma unroll 1
  for (unsigned ih_high = 0; ih_high < in_h + 2 * pad; ih_high += p::tile_h) {
#pragma unroll 1
    for (unsigned iw_high = 0; iw_high < in_w + 2 * pad; iw_high += p::tile_w) {
      input_load_module(in_data, in_buf, in_h, in_w, in_c_by_word, ih_high, iw_high, pad);

#pragma unroll 1
      for (unsigned oc_high = 0; oc_high < out_c; oc_high += out_c_low) {
        // threshold loading module.
        // just relay on automatic unroll.
        hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4, 5) T_out_hls threshold_buf[out_c_low][p::num_thresholds];

        kernel_load_module(k_data, k_buf, k_h, k_w, in_c_by_word, oc_high);

        if (use_threshold > 0) {
          threshold_load_module(threshold_data, threshold_buf, oc_high);
        }

        mac_compute_module(in_buf, out_buf, k_buf, threshold_buf, in_c_by_word, k_h, k_w);

        output_store_module(out_data, out_buf, threshold_buf, out_h, out_w, out_c, ih_high, iw_high, oc_high,
                            use_threshold);
      }
    }
  }
}
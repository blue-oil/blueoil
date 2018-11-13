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

hls_avalon_slave_component void intel_hls_qgemm_impl(
  hls_avalon_slave_register_argument
    ihc::mm_master<T_A_hls, ihc::aspace<1>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_IN>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true>> &A_packed,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_B_hls, ihc::aspace<2>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_K>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true>> &B_packed,
  hls_avalon_slave_register_argument
    ihc::mm_master<T_Y_hls, ihc::aspace<3>, ihc::awidth<32>, ihc::dwidth<NBITS_BW_OUT>, ihc::latency<0>,
                   ihc::maxburst<32>, ihc::align<32>, ihc::waitrequest<true>> &Y,
  hls_avalon_slave_register_argument uint32 a_row, hls_avalon_slave_register_argument uint32 a_col_by_word,
  hls_avalon_slave_register_argument uint32 b_col)
{
  static const unsigned gemm_max_a_col_by_word = 32 * 9;
  assert(gemm_max_a_col_by_word > p::max_in_c_by_word);
  unsigned idx_b = 0;

  for (unsigned bc_out = 0; bc_out < b_col; bc_out += p::num_pe) {
    hls_memory hls_singlepump hls_bankbits(0, 1, 2, 3, 4, 5, 6) T_B_hls B_local[gemm_max_a_col_by_word][p::num_pe];

#pragma unroll 4
    for (unsigned br = 0; br < a_col_by_word; br++) {
#pragma unroll
      for (unsigned bc_in = 0; bc_in < p::num_pe; bc_in++) { B_local[br][bc_in] = B_packed[idx_b++]; }
    }

    unsigned idx_y = bc_out;
    unsigned idx_a = 0;

    for (unsigned ar = 0; ar < a_row; ar++) {
#pragma unroll 8
      hls_register int16 y0[p::num_pe] = {0, 0, 0, 0, 0, 0, 0, 0};

      hls_register int16 y1[p::num_pe] = {0, 0, 0, 0, 0, 0, 0, 0};

      for (unsigned ac = 0; ac < a_col_by_word; ac++) {
        T_A_hls a0 = A_packed[idx_a++];
        T_A_hls a1 = A_packed[idx_a++];

#pragma unroll
        for (unsigned bc_in = 0; bc_in < p::num_pe; bc_in++) {
          T_B_hls b = B_local[ac][bc_in];
          int8 nk_pop = __builtin_popcount(~b);
          int8 xnor_pop0 = __builtin_popcount(~(a0 ^ b));
          int8 xnor_pop1 = __builtin_popcount(~(a1 ^ b));
          y0[bc_in] += (xnor_pop0 - nk_pop) + ((xnor_pop1 - nk_pop) << 1);
        }
      }

#pragma unroll
      for (unsigned bc_in = 0; bc_in < p::num_pe; bc_in++) { y1[bc_in] = y0[bc_in]; }
#pragma unroll
      for (unsigned bc_in = 0; bc_in < p::num_pe; bc_in++) { Y[idx_y + bc_in] = y1[bc_in]; }
      idx_y += b_col;
    }
  }
}

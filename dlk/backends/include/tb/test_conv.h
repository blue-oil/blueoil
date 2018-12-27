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
#include "common/global.h"
#include "common/utils.h"
#include "cpp.h"
#include "tb/test_input.h"

#if defined _INTEL_HLS_
#include "intel_hls.h"
#elif defined _DE10_NANO_
#include "de10_nano.h"
#endif

template <int KH, int KW>
bool test_conv(input_type &in_type, Conv_params_t &p)
{
  T_in *in_data = new T_in[p.in_size];
  T_in *in_data_packed = new T_in[p.in_size_packed];

  T_k *k_data = new T_k[p.k_size * p.k_n];
  T_k *k_data_with_kn2row = new T_k[p.k_size * p.k_n];
  T_q *k_data_packed = new T_q[p.k_size_packed * p.k_n];
  T_q *k_data_packed_t = new T_q[p.k_size_packed * p.k_n];
  T_q *k_data_packed_hwnocni = new T_q[p.k_size_packed * p.k_n];

  T_out *out_data = new T_out[p.out_size];
  T_out *out_data_gemm = new T_out[p.out_size];
  T_out *out_data_conv_kn2row_tiling = new T_out[p.out_size];
  T_out *out_data_packed = new T_out[p.out_size];
  T_out *out_data_with_kn2row = new T_out[p.out_size];
  T_out *out_data_qconv_kn2row_tiling = new T_out[p.out_size];
  T_out *out_data_hls = new T_out[p.out_size];
  T_out *out_data_hls_qgemm = new T_out[p.out_size];
  T_out *out_data_hls_qconv_kn2row_tiling = new T_out[p.out_size];
  T_out *out_data_fpga = new T_out[p.out_size];
  T_out *out_data_fpga_qkt = new T_out[p.out_size];

  T_out *threshold_data = NULL;

  bool comp_packed = true;
  bool comp_gemm = true;
  bool comp_hls = true;
  bool comp_fpga = true;

  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Conv" << p.k_h << "x" << p.k_w << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Layer info" << std::endl
            << "input: " << std::endl
            << "  height: " << p.in_h << std::endl
            << "  width: " << p.in_w << std::endl
            << "  channel: " << p.in_c << std::endl
            << "  packed channel: " << p.in_c_by_word << std::endl
            << "kernel: " << std::endl
            << "  height: " << p.k_h << std::endl
            << "  width: " << p.k_w << std::endl
            << "output: " << std::endl
            << "  height: " << p.out_h << std::endl
            << "  width: " << p.out_w << std::endl
            << "  channel: " << p.out_c << std::endl
            << "num_pe: " << p.num_pe << std::endl
            << "threshold_skipping: " << ((p.has_thresholds) ? "on" : "off") << std::endl
            << "tile: " << std::endl
            << "  height: " << conv_kn2row_params::tile_h << std::endl
            << "  width: " << conv_kn2row_params::tile_w << std::endl
            << "-------------------------------------------" << std::endl;

  if (in_type == SEQUENTIAL) {
    for (int i = 0; i < p.in_size; i++) { in_data[i] = (i % 4); }
    for (int i = 0; i < p.k_size * p.k_n; i++) { k_data[i] = (i % 2 == 0) ? 1 : -1; }
  } else if (in_type == RANDOM) {
    for (int i = 0; i < p.in_size; i++) { in_data[i] = gen_random_value<T_in>(4, 1, 0); }
    for (int i = 0; i < p.k_size * p.k_n; i++) { k_data[i] = gen_random_value<T_k>(2, 2, 1); }
  } else if (in_type == ALL_1) {
    for (int i = 0; i < p.in_size; i++) { in_data[i] = 1; }
    for (int i = 0; i < p.k_size * p.k_n; i++) { k_data[i] = 1; }
  }

  if (p.has_thresholds) {
    threshold_data = new T_out[p.out_c * p.num_thresholds];

    for (unsigned oc = 0; oc < p.out_c; oc++) {
      for (unsigned i = 0; i < p.num_thresholds; i++) {
        unsigned idx = oc * p.num_thresholds + i;
        if (i == 3) {
          // 1 or -1 means increasing or decreasing function
          threshold_data[idx] = gen_random_value<T_k>(2, 2, 1);
        } else {
          threshold_data[idx] = gen_random_value<T_out>(50, 1, 25);
        }
      }
    }
  }

  cpp::conv<KH, KW>(in_data, out_data, k_data, threshold_data, p.in_w, p.in_h, p.in_c, p.out_w, p.out_h, p.out_c,
                    p.pad_w, p.stride_w);

  kernel_transform_NHWC_to_NoHWCNi(k_data, k_data_with_kn2row, p.k_n, KH, KW, p.k_c, p.num_pe);
  cpp::conv_kn2row_tiling<KH, KW>(in_data, out_data_conv_kn2row_tiling, k_data_with_kn2row, threshold_data, p.in_w,
                                  p.in_h, p.in_c, p.out_w, p.out_h, p.out_c, p.pad_w, p.stride_w);
  comp_packed = compare_output(out_data_conv_kn2row_tiling, out_data, "conv_kn2row_tiling", p.out_h, p.out_w, p.out_c);

  pack_input_channel_wise(in_data, in_data_packed, p.in_h, p.in_w, p.in_c, p.nbits_in_data);

  pack_kernel_channel_wise(k_data, k_data_packed, p.k_h, p.k_w, p.k_c, p.k_n);

  kernel_transform_NHWC_to_NoHWCNi(k_data_packed, k_data_packed_t, p.k_n, p.k_h, p.k_w, p.k_c_by_word, p.num_pe);

  kernel_transform_NHWC_to_HWNoCNi(k_data_packed, k_data_packed_hwnocni, p.k_n, p.k_h, p.k_w, p.k_c_by_word, p.num_pe);

  // cpp::qconv_with_kn2row<KH, KW>(in_data_packed, out_data_with_kn2row, k_data_packed_hwnocni, threshold_data, p.in_w,
  //                                p.in_h, p.in_c_by_word, p.nbits_in_data, p.out_w, p.out_h, p.out_c, p.pad_w,
  //                                p.stride_w);
  // comp_packed = compare_output(out_data_with_kn2row, out_data, "qconv_with_kn2row", p.out_h, p.out_w, p.out_c);

  // cpp::qconv_kn2row_tiling<KH, KW>(in_data_packed, out_data_qconv_kn2row_tiling, k_data_packed_t, threshold_data,
  //                                  p.in_w, p.in_h, p.in_c_by_word, p.nbits_in_data, p.out_w, p.out_h, p.out_c,
  //                                  p.pad_w, p.stride_w);
  // comp_packed =
  //   compare_output(out_data_qconv_kn2row_tiling, out_data, "qconv_kn2row_tiling", p.out_h, p.out_w, p.out_c);

#if defined _INTEL_HLS_

  // intel_hls_qconv_with_kn2row(in_data_packed, out_data_hls_qgemm, k_data_packed_hwnocni, threshold_data, p.in_w,
  // p.in_h,
  //                             p.in_c_by_word, p.nbits_in_data, p.out_w, p.out_h, p.out_c, p.k_w, p.k_h, p.pad_w,
  //                             p.stride_w);
  // comp_packed = compare_output(out_data_hls_qgemm, out_data, "hls_qgemm", p.out_h, p.out_w, p.out_c);

  intel_hls_qconv_kn2row_tiling(in_data_packed, out_data_hls_qconv_kn2row_tiling, k_data_packed_t, threshold_data,
                                p.in_w, p.in_h, p.in_c_by_word, p.nbits_in_data, p.out_w, p.out_h, p.out_c, p.k_w,
                                p.k_h, p.pad_w, p.stride_w);
  comp_packed =
    compare_output(out_data_hls_qconv_kn2row_tiling, out_data, "hls_qconv_kn2row_tiling", p.out_h, p.out_w, p.out_c);

#elif defined _DE10_NANO_

  // de10_nano::qconv_with_kn2row(p.k_w, p.k_h, in_data_packed, out_data_fpga, k_data_packed_hwnocni, p.in_w, p.in_h,
  //                              p.in_c_by_word, p.nbits_in_data, p.out_w, p.out_h, p.out_c, p.pad_w, p.stride_w);
  // comp_fpga = compare_output(out_data_fpga, out_data, "qconv_with_kn2row_fpga", p.out_h, p.out_w, p.out_c);

  de10_nano::qconv_kn2row_tiling(p.k_w, p.k_h, in_data_packed, out_data_fpga_qkt, k_data_packed_t, threshold_data,
                                 p.in_w, p.in_h, p.in_c_by_word, p.nbits_in_data, p.out_w, p.out_h, p.out_c, p.pad_w,
                                 p.stride_w);
  comp_fpga = compare_output(out_data_fpga_qkt, out_data, "qconv_kn2row_tiling_fpga", p.out_h, p.out_w, p.out_c);

#endif

  delete[] in_data;
  delete[] in_data_packed;

  delete[] k_data;
  delete[] k_data_packed;
  delete[] k_data_packed_t;

  delete[] out_data;
  delete[] out_data_gemm;
  delete[] out_data_packed;
  delete[] out_data_with_kn2row;
  delete[] out_data_hls;
  delete[] out_data_hls_qgemm;
  delete[] out_data_fpga;

  return comp_packed && comp_hls && comp_fpga;
}

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
bool test_a8w1_conv(input_type &in_type)
{
  namespace p = a8w1_conv3x3_params;

  assert(p::k_h == KH);
  assert(p::k_w == KW);

  T_in_k2c *in_data = new T_in_k2c[p::in_size];
  T_k_k2c *k_data = new T_k_k2c[p::k_size * p::k_n];
  T_k_k2c *k_data_hwcn = new T_k_k2c[p::k_size * p::k_n];
  T_k_k2c *k_data_hwnocni = new T_k_k2c[p::k_size * p::k_n];
  T_out_k2c *out_data = new T_out_k2c[p::out_size];
  T_out_k2c *out_data_kn2row = new T_out_k2c[p::out_size];
  T_out_k2c *out_data_hls = new T_out_k2c[p::out_size];
  T_out_k2c *out_data_fpga = new T_out_k2c[p::out_size];

  bool comp_hls = true;
  bool comp_fpga = true;

  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "A8W1 Conv" << p::k_h << "x" << p::k_w << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Layer info" << std::endl
            << "input: " << std::endl
            << "  height: " << p::in_h << std::endl
            << "  width: " << p::in_w << std::endl
            << "  channel: " << p::in_c << std::endl
            << "  packed channel: " << p::in_c << std::endl
            << "kernel: " << std::endl
            << "  height: " << p::k_h << std::endl
            << "  width: " << p::k_w << std::endl
            << "output: " << std::endl
            << "  height: " << p::out_h << std::endl
            << "  width: " << p::out_w << std::endl
            << "  channel: " << p::out_c << std::endl
            << "threhsold_skipping: " << ((p::has_thresholds) ? "on" : "off") << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  if (in_type == SEQUENTIAL) {
    for (int i = 0; i < p::in_size; i++) { in_data[i] = (i % 4); }
    for (int i = 0; i < p::k_size * p::k_n; i++) { k_data[i] = (i % 2 == 0) ? 1 : -1; }
  } else if (in_type == RANDOM) {
    for (int i = 0; i < p::in_size; i++) { in_data[i] = gen_random_value<T_in>(256, 1, 0); }
    for (int i = 0; i < p::k_size * p::k_n; i++) { k_data[i] = gen_random_value<T_k>(2, 2, 1); }
  } else if (in_type == ALL_1) {
    for (int i = 0; i < p::in_size; i++) { in_data[i] = 1; }
    for (int i = 0; i < p::k_size * p::k_n; i++) { k_data[i] = 1; }
  }

  cpp::a8w1_conv<KH, KW>(in_data, out_data, k_data, p::in_w, p::in_h, p::in_c, p::out_w, p::out_h, p::out_c, p::pad_w,
                         p::stride_w);

  kernel_transform_NHWC_to_HWCN(k_data, k_data_hwcn, p::k_n, p::k_h, p::k_w, p::k_c);

  cpp::a8w1_conv3x3_with_kn2row<KH, KW>(in_data, out_data_kn2row, k_data_hwcn, p::in_w, p::in_h, p::in_c, p::out_w,
                                        p::out_h, p::out_c, p::pad_w, p::stride_w);

  bool comp_kn2row = compare_output(out_data_kn2row, out_data, "a8w1_qconv_kn2row", p::out_h, p::out_w, p::out_c);

  kernel_transform_NHWC_to_HWNoCNi(k_data, k_data_hwnocni, p::k_n, p::k_h, p::k_w, p::k_c, p::num_pe);

#if defined _INTEL_HLS_

  intel_hls_a8w1_qconv_with_kn2row(in_data, out_data_hls, k_data_hwnocni, p::in_w, p::in_h, p::in_c, p::out_w, p::out_h,
                                   p::out_c, p::k_w, p::k_h, p::pad_w, p::stride_w);
  comp_hls = compare_output(out_data_hls, out_data, "a8w1_qconv_kn2row_hls", p::out_h, p::out_w, p::out_c);

#elif defined _DE10_NANO_

  de10_nano::A8W1_qconv(in_data, out_data_fpga, k_data_hwnocni, p::in_w, p::in_h, p::in_c, p::out_w, p::out_h, p::out_c,
                        p::k_w, p::k_h, p::pad_w, p::stride_w);

  comp_fpga = compare_output(out_data_fpga, out_data, "fpga", p::out_h, p::out_w, p::out_c);

  // unsigned idx_out = 0;
  // for(unsigned oh = 0; oh < p::out_h; oh++) {
  // for(unsigned ow = 0; ow < p::out_w; ow++) {
  // for(unsigned oc = 0; oc < p::out_c; oc++) {
  //   printf("h: %u, w: %u, c: %u, out: %d, ex: %d\n",
  //           oh, ow, oc, out_data_fpga[idx_out], out_data[idx_out]);
  //   idx_out++;
  // }
  // }
  // }

#endif

  delete[] in_data;
  delete[] k_data;
  delete[] out_data;
  delete[] out_data_hls;

  return comp_kn2row;
}

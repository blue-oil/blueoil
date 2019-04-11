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
#include "global.h"
#include "memdriver.h"
#include "time_measurement.h"
#include <cassert>

namespace de10_nano {

class QconvWithKn2row {
public:
  QconvWithKn2row()
      : start(IP_CSR_ADDR + 0x08, 1, sizeof(T_UINT)),
        done(IP_CSR_ADDR + 0x18, 1, sizeof(T_UINT)),
        in_data_reg(IP_CSR_ADDR + 0x20, 1, sizeof(T_UINT)),
        out_data_reg(IP_CSR_ADDR + 0x28, 1, sizeof(T_UINT)),
        k_data_reg(IP_CSR_ADDR + 0x30, 1, sizeof(T_UINT)),
        out_data_partial_reg(IP_CSR_ADDR + 0x38, 1, sizeof(T_UINT)),
        in_w_reg(IP_CSR_ADDR + 0x40, 1, sizeof(T_UINT)),
        in_h_reg(IP_CSR_ADDR + 0x48, 1, sizeof(T_UINT)),
        in_c_reg(IP_CSR_ADDR + 0x50, 1, sizeof(T_UINT)),
        out_w_reg(IP_CSR_ADDR + 0x58, 1, sizeof(T_UINT)),
        out_h_reg(IP_CSR_ADDR + 0x60, 1, sizeof(T_UINT)),
        out_c_reg(IP_CSR_ADDR + 0x68, 1, sizeof(T_UINT)),
        k_w_reg(IP_CSR_ADDR + 0x70, 1, sizeof(T_UINT)),
        k_h_reg(IP_CSR_ADDR + 0x78, 1, sizeof(T_UINT)),
        pad_reg(IP_CSR_ADDR + 0x80, 1, sizeof(T_UINT)),

        th_start(TH_IP_CSR_ADDR + 0x08, 1, sizeof(T_UINT)),
        th_done(TH_IP_CSR_ADDR + 0x18, 1, sizeof(T_UINT)),
        th_in_data_reg(TH_IP_CSR_ADDR + 0x20, 1, sizeof(T_UINT)),
        th_out_data_reg(TH_IP_CSR_ADDR + 0x28, 1, sizeof(T_UINT)),
        th_data_reg(TH_IP_CSR_ADDR + 0x30, 1, sizeof(T_UINT)),
        th_out_w_reg(TH_IP_CSR_ADDR + 0x38, 1, sizeof(T_UINT)),
        th_out_h_reg(TH_IP_CSR_ADDR + 0x40, 1, sizeof(T_UINT)),
        th_out_c_reg(TH_IP_CSR_ADDR + 0x48, 1, sizeof(T_UINT)) {}

  void run_qconv_with_kn2row(unsigned long in_data_addr,
                             unsigned long out_data_addr,
                             unsigned long k_data_addr, unsigned in_w,
                             unsigned in_h, unsigned in_c, unsigned out_w,
                             unsigned out_h, unsigned out_c, unsigned k_w,
                             unsigned k_h, unsigned pad) {
    in_data_reg.Write(in_data_addr);
    out_data_reg.Write(out_data_addr);
    k_data_reg.Write(k_data_addr);
    out_data_partial_reg.Write(out_data_addr);

    in_w_reg.Write(in_w);
    in_h_reg.Write(in_h);
    in_c_reg.Write(in_c);
    out_w_reg.Write(out_w);
    out_h_reg.Write(out_h);
    out_c_reg.Write(out_c);
    k_h_reg.Write(k_h);
    k_w_reg.Write(k_w);
    pad_reg.Write(pad);

    start.Write(0x1);

    volatile T_UINT done_flag = 0;
    while (!(done_flag & 0x2)) {
      done.Read(done_flag);
    }
  }

  void run_apply_thresholds(unsigned long in_data_addr,
                            unsigned long out_data_addr,
                            unsigned long th_data_addr, unsigned out_w,
                            unsigned out_h, unsigned out_c) {
    th_in_data_reg.Write(in_data_addr);
    th_out_data_reg.Write(out_data_addr);
    th_data_reg.Write(th_data_addr);

    th_out_w_reg.Write(out_w);
    th_out_h_reg.Write(out_h);
    th_out_c_reg.Write(out_c);

    th_start.Write(0x1);

    volatile T_UINT done_flag = 0;
    while (!(done_flag & 0x2)) {
      th_done.Read(done_flag);
    }
  }

private:
  MappedMem start;
  MappedMem done;
  MappedMem in_data_reg;
  MappedMem out_data_reg;
  MappedMem k_data_reg;
  MappedMem out_data_partial_reg;
  MappedMem in_w_reg;
  MappedMem in_h_reg;
  MappedMem in_c_reg;
  MappedMem out_w_reg;
  MappedMem out_h_reg;
  MappedMem out_c_reg;
  MappedMem k_w_reg;
  MappedMem k_h_reg;
  MappedMem pad_reg;

  MappedMem th_start;
  MappedMem th_done;
  MappedMem th_in_data_reg;
  MappedMem th_out_data_reg;
  MappedMem th_data_reg;
  MappedMem th_out_w_reg;
  MappedMem th_out_h_reg;
  MappedMem th_out_c_reg;
};

void qconv_with_kn2row(unsigned long input_addr, unsigned long output_addr,
                       const QUANTIZED_PACKED_KERNEL k_data_packed[], BIN_CONV_OUTPUT th_data[],
                       unsigned in_w, unsigned in_h, unsigned in_c_by_word,
                       unsigned nbits_in_data, unsigned out_w, unsigned out_h,
                       unsigned out_c, unsigned k_w, unsigned k_h, unsigned pad,
                       unsigned stride) {
  assert((k_h == 1 && k_w == 1) || (k_h == 3 && k_w == 3));

  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;
  static QconvWithKn2row qwq;

  MappedMem k_data_mem(KERNEL_ADDR, k_size, sizeof(T_UINT));
  k_data_mem.Write(k_data_packed, k_size);

  if (th_data == nullptr) {
    qwq.run_qconv_with_kn2row(input_addr, output_addr, KERNEL_ADDR, in_w, in_h,
                              in_c_by_word, out_w, out_h, out_c, k_w, k_h, pad);
  } else { // with threshold skipping
    const unsigned num_th = NUM_OF_A2W1_THRESHOLD;
    const unsigned th_size = out_c * num_th;

    MappedMem th_data_mem(THRESHOLD_ADDR, th_size, sizeof(BIN_CONV_OUTPUT));
    th_data_mem.Write(th_data, th_size);

    qwq.run_qconv_with_kn2row(input_addr, OUTPUT1_ADDR, KERNEL_ADDR, in_w, in_h,
                              in_c_by_word, out_w, out_h, out_c, k_w, k_h, pad);

    qwq.run_apply_thresholds(OUTPUT1_ADDR, output_addr, THRESHOLD_ADDR, out_w,
                             out_h, out_c);
  }
}

class QconvKn2rowTiling {
public:
  QconvKn2rowTiling()
      : start(IP_CSR_ADDR + 0x08, 1, sizeof(T_UINT)),
        done(IP_CSR_ADDR + 0x18, 1, sizeof(T_UINT)),
        in_data_reg(IP_CSR_ADDR + 0x20, 1, sizeof(T_UINT)),
        out_data_reg(IP_CSR_ADDR + 0x28, 1, sizeof(T_UINT)),
        k_data_reg(IP_CSR_ADDR + 0x30, 1, sizeof(T_UINT)),
        th_data_reg(IP_CSR_ADDR + 0x38, 1, sizeof(T_UINT)),
        in_w_reg(IP_CSR_ADDR + 0x40, 1, sizeof(T_UINT)),
        in_h_reg(IP_CSR_ADDR + 0x48, 1, sizeof(T_UINT)),
        in_c_reg(IP_CSR_ADDR + 0x50, 1, sizeof(T_UINT)),
        out_w_reg(IP_CSR_ADDR + 0x58, 1, sizeof(T_UINT)),
        out_h_reg(IP_CSR_ADDR + 0x60, 1, sizeof(T_UINT)),
        out_c_reg(IP_CSR_ADDR + 0x68, 1, sizeof(T_UINT)),
        k_w_reg(IP_CSR_ADDR + 0x70, 1, sizeof(T_UINT)),
        k_h_reg(IP_CSR_ADDR + 0x78, 1, sizeof(T_UINT)),
        pad_reg(IP_CSR_ADDR + 0x80, 1, sizeof(T_UINT)),
        use_threshold_reg(IP_CSR_ADDR + 0x88, 1, sizeof(T_UINT)) {}

  void run(unsigned long in_data_addr, unsigned long out_data_addr,
           unsigned long k_data_addr, unsigned long th_data_addr, unsigned in_w,
           unsigned in_h, unsigned in_c, unsigned out_w, unsigned out_h,
           unsigned out_c, unsigned k_w, unsigned k_h, unsigned pad,
           unsigned use_threshold) {
    in_data_reg.Write(in_data_addr);
    out_data_reg.Write(out_data_addr);
    k_data_reg.Write(k_data_addr);
    th_data_reg.Write(th_data_addr);

    in_w_reg.Write(in_w);
    in_h_reg.Write(in_h);
    in_c_reg.Write(in_c);
    out_w_reg.Write(out_w);
    out_h_reg.Write(out_h);
    out_c_reg.Write(out_c);
    k_h_reg.Write(k_h);
    k_w_reg.Write(k_w);
    pad_reg.Write(pad);
    use_threshold_reg.Write(use_threshold);

    start.Write(0x1);

    volatile T_UINT done_flag = 0;
    while (!(done_flag & 0x2)) {
      done.Read(done_flag);
    }
  }

private:
  MappedMem start;
  MappedMem done;
  MappedMem in_data_reg;
  MappedMem out_data_reg;
  MappedMem k_data_reg;
  MappedMem th_data_reg;
  MappedMem in_w_reg;
  MappedMem in_h_reg;
  MappedMem in_c_reg;
  MappedMem out_w_reg;
  MappedMem out_h_reg;
  MappedMem out_c_reg;
  MappedMem k_w_reg;
  MappedMem k_h_reg;
  MappedMem pad_reg;
  MappedMem use_threshold_reg;
};

void qconv_kn2row_tiling(unsigned long input_addr, unsigned long output_addr,
                         const QUANTIZED_PACKED_KERNEL k_data_packed[],
                         BIN_CONV_OUTPUT th_data[], unsigned in_w,
                         unsigned in_h, unsigned in_c_by_word,
                         unsigned nbits_in_data, unsigned out_w, unsigned out_h,
                         unsigned out_c, unsigned k_w, unsigned k_h,
                         unsigned pad, unsigned stride) {
  assert((k_h == 1 && k_w == 1) || (k_h == 3 && k_w == 3));

  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;

  static QconvKn2rowTiling qkt;
  MappedMem k_data_mem(KERNEL_ADDR, k_size, sizeof(QUANTIZED_PACKED_KERNEL));
  k_data_mem.Write(k_data_packed, k_size);
  unsigned use_threshold = (th_data != NULL) ? 1 : 0;

  if (use_threshold == 1) {
    const unsigned th_size = out_c * NUM_OF_A2W1_THRESHOLD;
    MappedMem th_data_mem(THRESHOLD_ADDR, th_size, sizeof(BIN_CONV_OUTPUT));
    th_data_mem.Write(th_data, th_size);

    qkt.run(input_addr, output_addr, KERNEL_ADDR, THRESHOLD_ADDR, in_w, in_h,
            in_c_by_word, out_w, out_h, out_c, k_w, k_h, pad, use_threshold);
  } else {
    qkt.run(input_addr, output_addr, KERNEL_ADDR, THRESHOLD_ADDR, in_w, in_h,
            in_c_by_word, out_w, out_h, out_c, k_w, k_h, pad, use_threshold);
  }
}

} // namespace de10_nano

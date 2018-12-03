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
#include <chrono>
#include "fpga_utils.h"

using namespace std::chrono;
using std::cout;
using std::endl;

namespace de10_nano {
using namespace type;

class FPGA
{
public:
  FPGA()
    : start(IP_CSR_ADDR + 0x08, 1, sizeof(u32))
    , done(IP_CSR_ADDR + 0x18, 1, sizeof(u32))
    , in_data_reg(IP_CSR_ADDR + 0x20, 1, sizeof(u32))
    , out_data_reg(IP_CSR_ADDR + 0x28, 1, sizeof(u32))
    , k_data_reg(IP_CSR_ADDR + 0x30, 1, sizeof(u32))
    , in_w_reg(IP_CSR_ADDR + 0x38, 1, sizeof(u32))
    , in_h_reg(IP_CSR_ADDR + 0x40, 1, sizeof(u32))
    , in_c_reg(IP_CSR_ADDR + 0x48, 1, sizeof(u32))
    , out_w_reg(IP_CSR_ADDR + 0x50, 1, sizeof(u32))
    , out_h_reg(IP_CSR_ADDR + 0x58, 1, sizeof(u32))
    , out_c_reg(IP_CSR_ADDR + 0x60, 1, sizeof(u32))
    , out_c_offset_reg(IP_CSR_ADDR + 0x68, 1, sizeof(u32))
  {}

  void conv3x3(unsigned in_w, unsigned in_h, unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c,
               unsigned out_c_offset)
  {
    in_data_reg.Write(IN_DATA_ADDR);
    out_data_reg.Write(OUT_DATA_ADDR);
    k_data_reg.Write(K_DATA_ADDR);

    in_w_reg.Write(in_w);
    in_h_reg.Write(in_h);
    in_c_reg.Write(in_c);
    out_w_reg.Write(out_w);
    out_h_reg.Write(out_h);
    out_c_reg.Write(out_c);
    out_c_offset_reg.Write(out_c_offset);

    start.Write(0x1);

    volatile u32 done_flag = 0;
    while (!(done_flag & 0x2)) { done.Read(done_flag); }
  }

  void conv1x1(unsigned in_w, unsigned in_h, unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c,
               unsigned out_c_offset)
  {
    in_data_reg.Write(IN_DATA_ADDR);
    out_data_reg.Write(OUT_DATA_ADDR);
    k_data_reg.Write(K_DATA_ADDR);

    in_w_reg.Write(in_w);
    in_h_reg.Write(in_h);
    in_c_reg.Write(in_c);
    out_w_reg.Write(out_w);
    out_h_reg.Write(out_h);
    out_c_reg.Write(out_c);
    out_c_offset_reg.Write(out_c_offset);

    start.Write(0x1);

    volatile u32 done_flag = 0;
    while (!(done_flag & 0x2)) { done.Read(done_flag); }
  }

private:
  MappedMem start;
  MappedMem done;
  MappedMem in_data_reg;
  MappedMem out_data_reg;
  MappedMem k_data_reg;
  MappedMem in_w_reg;
  MappedMem in_h_reg;
  MappedMem in_c_reg;
  MappedMem out_w_reg;
  MappedMem out_h_reg;
  MappedMem out_c_reg;
  MappedMem out_c_offset_reg;
};

void qconv(unsigned k_w, unsigned k_h, T_q in_data_packed[], T_out out_data[], T_q k_data_packed[], unsigned in_w,
           unsigned in_h, unsigned in_c_by_word, unsigned nbits_in_data, unsigned out_w, unsigned out_h, unsigned out_c,
           unsigned pad, unsigned stride)
{
  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;

  static FPGA fpga;
  MappedMem in_data_mem(IN_DATA_ADDR, in_size, sizeof(T_in));
  MappedMem out_data_mem(OUT_DATA_ADDR, out_size, sizeof(T_out));
  MappedMem k_data_mem(K_DATA_ADDR, k_size, sizeof(T_k));

  if (k_h == 3 && k_w == 3) {
  } else if (k_h == 1 && k_w == 1) {
    in_data_mem.Write(in_data_packed, in_size);
    k_data_mem.Write(k_data_packed, k_size);

    auto start = system_clock::now();
    fpga.conv1x1(in_w, in_h, in_c_by_word, out_w, out_h, out_c, 0);
    auto end = system_clock::now();

    out_data_mem.Read(out_data, out_size);

    auto diff = end - start;
    cout << "FPGA exec time: " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " [msec]"
         << std::endl;

    std::cout << "out_size: " << out_size << std::endl;
  } else {
    std::cout << "conv" << k_h << "x" << k_w << "is not supported..." << std::endl;
  }
}

class QconvWithQgemm
{
public:
  QconvWithQgemm()
    : start(IP_CSR_ADDR + 0x08, 1, sizeof(u32))
    , done(IP_CSR_ADDR + 0x18, 1, sizeof(u32))
    , in_data_reg(IP_CSR_ADDR + 0x20, 1, sizeof(u32))
    , out_data_reg(IP_CSR_ADDR + 0x28, 1, sizeof(u32))
    , k_data_reg(IP_CSR_ADDR + 0x30, 1, sizeof(u32))
    , out_data_partial_reg(IP_CSR_ADDR + 0x38, 1, sizeof(u32))
    , in_w_reg(IP_CSR_ADDR + 0x40, 1, sizeof(u32))
    , in_h_reg(IP_CSR_ADDR + 0x48, 1, sizeof(u32))
    , in_c_reg(IP_CSR_ADDR + 0x50, 1, sizeof(u32))
    , out_w_reg(IP_CSR_ADDR + 0x58, 1, sizeof(u32))
    , out_h_reg(IP_CSR_ADDR + 0x60, 1, sizeof(u32))
    , out_c_reg(IP_CSR_ADDR + 0x68, 1, sizeof(u32))
    , k_w_reg(IP_CSR_ADDR + 0x70, 1, sizeof(u32))
    , k_h_reg(IP_CSR_ADDR + 0x78, 1, sizeof(u32))
    , pad_reg(IP_CSR_ADDR + 0x80, 1, sizeof(u32))
  {}

  void run(unsigned in_w, unsigned in_h, unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w,
           unsigned k_h, unsigned pad)
  {
    in_data_reg.Write(IN_DATA_ADDR);
    out_data_reg.Write(OUT_DATA_ADDR);
    k_data_reg.Write(K_DATA_ADDR);
    out_data_partial_reg.Write(OUT_DATA_ADDR);

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

    volatile u32 done_flag = 0;
    while (!(done_flag & 0x2)) { done.Read(done_flag); }
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
};

void qconv_with_kn2row(unsigned k_w, unsigned k_h, T_q in_data_packed[], T_out out_data[], T_q k_data_packed[],
                       unsigned in_w, unsigned in_h, unsigned in_c_by_word, unsigned nbits_in_data, unsigned out_w,
                       unsigned out_h, unsigned out_c, unsigned pad, unsigned stride)
{
  assert((k_h == 1 && k_w == 1) || (k_h == 3 && k_w == 3));

  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;

  static QconvWithQgemm qwq;
  MappedMem in_data_mem(IN_DATA_ADDR, in_size, sizeof(T_in));
  MappedMem out_data_mem(OUT_DATA_ADDR, out_size, sizeof(T_out));
  MappedMem k_data_mem(K_DATA_ADDR, k_size, sizeof(T_k));

  in_data_mem.Write(in_data_packed, in_size);
  k_data_mem.Write(k_data_packed, k_size);

  auto start = system_clock::now();
  qwq.run(in_w, in_h, in_c_by_word, out_w, out_h, out_c, k_w, k_h, pad);
  auto end = system_clock::now();

  out_data_mem.Read(out_data, out_size);

  auto diff = end - start;
  cout << "FPGA exec time: " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " [msec]"
       << std::endl;

  std::cout << "out_size: " << out_size << std::endl;
}

class QconvKn2rowTiling
{
public:
  QconvKn2rowTiling()
    : start(IP_CSR_ADDR + 0x08, 1, sizeof(u32))
    , done(IP_CSR_ADDR + 0x18, 1, sizeof(u32))
    , in_data_reg(IP_CSR_ADDR + 0x20, 1, sizeof(u32))
    , out_data_reg(IP_CSR_ADDR + 0x28, 1, sizeof(u32))
    , k_data_reg(IP_CSR_ADDR + 0x30, 1, sizeof(u32))
    , th_data_reg(IP_CSR_ADDR + 0x38, 1, sizeof(u32))
    , in_w_reg(IP_CSR_ADDR + 0x40, 1, sizeof(u32))
    , in_h_reg(IP_CSR_ADDR + 0x48, 1, sizeof(u32))
    , in_c_reg(IP_CSR_ADDR + 0x50, 1, sizeof(u32))
    , out_w_reg(IP_CSR_ADDR + 0x58, 1, sizeof(u32))
    , out_h_reg(IP_CSR_ADDR + 0x60, 1, sizeof(u32))
    , out_c_reg(IP_CSR_ADDR + 0x68, 1, sizeof(u32))
    , k_w_reg(IP_CSR_ADDR + 0x70, 1, sizeof(u32))
    , k_h_reg(IP_CSR_ADDR + 0x78, 1, sizeof(u32))
    , pad_reg(IP_CSR_ADDR + 0x80, 1, sizeof(u32))
    , use_threshold_reg(IP_CSR_ADDR + 0x88, 1, sizeof(u32))
  {}

  void run(unsigned in_w, unsigned in_h, unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w,
           unsigned k_h, unsigned pad, unsigned use_threshold)
  {
    in_data_reg.Write(IN_DATA_ADDR);
    out_data_reg.Write(OUT_DATA_ADDR);
    k_data_reg.Write(K_DATA_ADDR);
    th_data_reg.Write(THRESHOLDS_ADDR);

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

    volatile u32 done_flag = 0;
    while (!(done_flag & 0x2)) { done.Read(done_flag); }
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

void qconv_kn2row_tiling(unsigned k_w, unsigned k_h, T_q in_data_packed[], T_out out_data[], T_q k_data_packed[],
                         T_out th_data[], unsigned in_w, unsigned in_h, unsigned in_c_by_word, unsigned nbits_in_data,
                         unsigned out_w, unsigned out_h, unsigned out_c, unsigned pad, unsigned stride)
{
  assert((k_h == 1 && k_w == 1) || (k_h == 3 && k_w == 3));

  const unsigned in_size = in_h * in_w * in_c_by_word * nbits_in_data;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size = k_h * k_w * in_c_by_word * out_c;
  const unsigned th_size = out_c * conv_common_params::num_thresholds;

  static QconvKn2rowTiling qkt;
  MappedMem in_data_mem(IN_DATA_ADDR, in_size, sizeof(T_in));
  MappedMem out_data_mem(OUT_DATA_ADDR, out_size, sizeof(T_out));
  MappedMem k_data_mem(K_DATA_ADDR, k_size, sizeof(T_k));
  MappedMem th_data_mem(THRESHOLDS_ADDR, th_size, sizeof(T_out));

  in_data_mem.Write(in_data_packed, in_size);
  k_data_mem.Write(k_data_packed, k_size);
  th_data_mem.Write(th_data, th_size);

  unsigned use_threshold = (th_data != NULL) ? 1 : 0;

  auto start = system_clock::now();
  qkt.run(in_w, in_h, in_c_by_word, out_w, out_h, out_c, k_w, k_h, pad, use_threshold);
  auto end = system_clock::now();

  out_data_mem.Read(out_data, out_size);

  auto diff = end - start;
  cout << "FPGA exec time: " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " [msec]"
       << std::endl;

  std::cout << "out_size: " << out_size << std::endl;
}

class A8W1_Qconv
{
public:
  A8W1_Qconv()
    : start(A8W1_IP_CSR_ADDR + 0x08, 1, sizeof(u32))
    , done(A8W1_IP_CSR_ADDR + 0x18, 1, sizeof(u32))
    , in_data_reg(A8W1_IP_CSR_ADDR + 0x20, 1, sizeof(u32))
    , out_data_reg(A8W1_IP_CSR_ADDR + 0x28, 1, sizeof(u32))
    , k_data_reg(A8W1_IP_CSR_ADDR + 0x30, 1, sizeof(u32))
    , out_data_partial_reg(A8W1_IP_CSR_ADDR + 0x38, 1, sizeof(u32))
    , in_w_reg(A8W1_IP_CSR_ADDR + 0x40, 1, sizeof(u32))
    , in_h_reg(A8W1_IP_CSR_ADDR + 0x48, 1, sizeof(u32))
    , in_c_reg(A8W1_IP_CSR_ADDR + 0x50, 1, sizeof(u32))
    , out_w_reg(A8W1_IP_CSR_ADDR + 0x58, 1, sizeof(u32))
    , out_h_reg(A8W1_IP_CSR_ADDR + 0x60, 1, sizeof(u32))
    , out_c_reg(A8W1_IP_CSR_ADDR + 0x68, 1, sizeof(u32))
    , kw_reg(A8W1_IP_CSR_ADDR + 0x70, 1, sizeof(u32))
    , kh_reg(A8W1_IP_CSR_ADDR + 0x78, 1, sizeof(u32))
    , pad_reg(A8W1_IP_CSR_ADDR + 0x80, 1, sizeof(u32))
  {}

  void run(unsigned out0_data_addr, unsigned out1_data_addr, unsigned k_data_addr, unsigned in_w, unsigned in_h,
           unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c, unsigned kw, unsigned kh, unsigned pad)
  {
    in_data_reg.Write(IN_DATA_ADDR);
    out_data_reg.Write(out0_data_addr);
    k_data_reg.Write(k_data_addr);
    out_data_partial_reg.Write(out1_data_addr);

    in_w_reg.Write(in_w);
    in_h_reg.Write(in_h);
    in_c_reg.Write(in_c);
    out_w_reg.Write(out_w);
    out_h_reg.Write(out_h);
    out_c_reg.Write(out_c);
    kh_reg.Write(kh);
    kw_reg.Write(kw);
    pad_reg.Write(pad);

    start.Write(0x1);

    volatile u32 done_flag = 0;
    while (!(done_flag & 0x2)) { done.Read(done_flag); }
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
  MappedMem kw_reg;
  MappedMem kh_reg;
  MappedMem pad_reg;
};

void A8W1_qconv(T_in_k2c in_data[], T_out_k2c out_data[], T_k_k2c k_data[], unsigned in_w, unsigned in_h, unsigned in_c,
                unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w, unsigned k_h, unsigned pad,
                unsigned stride)
{
  assert((k_h == 1 && k_w == 1) || (k_h == 3 && k_w == 3));

  const unsigned in_size = in_h * in_w * in_c;
  const unsigned out_size = out_h * out_w * out_c;
  const unsigned k_size_partial = in_c * out_c;
  const unsigned k_size = k_h * k_h * k_size_partial;

  static A8W1_Qconv a8w1_qconv;
  MappedMem in_data_mem(IN_DATA_ADDR, in_size, sizeof(T_in));
  MappedMem k_data_mem(K_DATA_ADDR, k_size, sizeof(T_k));

  in_data_mem.Write(in_data, in_size);
  k_data_mem.Write(k_data, k_size);

  auto start = system_clock::now();

  for (unsigned char kh = 0; kh < k_h; kh++) {
    for (unsigned char kw = 0; kw < k_w; kw++) {
      const unsigned k_offset = kh * k_w * k_size_partial + kw * k_size_partial;
      const unsigned k_addr = K_DATA_ADDR + k_offset * sizeof(T_k_k2c);

      if ((kh * k_w + kw) % 2 == 0) {
        a8w1_qconv.run(OUT0_DATA_ADDR, OUT1_DATA_ADDR, k_addr, in_w, in_h, in_c, out_w, out_h, out_c, kw, kh, pad);
      } else {
        a8w1_qconv.run(OUT1_DATA_ADDR, OUT0_DATA_ADDR, k_addr, in_w, in_h, in_c, out_w, out_h, out_c, kw, kh, pad);
      }
    }
  }

  auto end = system_clock::now();

  MappedMem out0_data_mem(OUT0_DATA_ADDR, out_size, sizeof(T_out));
  MappedMem out1_data_mem(OUT1_DATA_ADDR, out_size, sizeof(T_out));

  if ((k_h * k_w) % 2 == 0) {
    out1_data_mem.Read(out_data, out_size);
  } else {
    out0_data_mem.Read(out_data, out_size);
  }

  auto diff = end - start;
  cout << "FPGA exec time: " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " [msec]"
       << std::endl;

  std::cout << "out_size: " << out_size << std::endl;
}
} // namespace de10_nano

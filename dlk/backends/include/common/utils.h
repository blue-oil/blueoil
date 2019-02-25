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
#include <cstdio>
#include <cstdlib>

using std::cout;
using std::endl;
using std::string;

#define NBITS(T) (sizeof(T) * 8)

unsigned pack_kernel_channel_wise(T_k k_data[], T_q k_data_packed[], int k_h, int k_w, int k_c, int k_n)
{
  namespace p = conv1x1_params;

  unsigned idx_out = 0;
  unsigned idx_k = 0;

  for (int i = 0; i < k_n; i++)
    for (int j = 0; j < k_h * k_w; j++)
      for (int l = 0; l < k_c / p::nbits_per_word; l++) {
        T_q out = 0;

        for (int bw = 0; bw < p::nbits_per_word; bw++) {
          T_k k = k_data[idx_k++];
          T_q k_packed = (k > 0) ? 1 : 0;
          k_packed = k_packed & 0x1;
          k_packed = k_packed << bw;
          out |= k_packed;
        }

        k_data_packed[idx_out++] = out;
      }

  return idx_out;
}

unsigned pack_input_channel_wise(T_in in_data[], T_q in_data_packed[], int in_h, int in_w, int in_c,
                                 unsigned nbits_in_data)
{
  namespace p = conv1x1_params;

  const unsigned nbits_per_word = sizeof(T_in) * 8;
  const unsigned in_c_by_word = in_c / (sizeof(T_in) * 8);
  unsigned idx_out = 0;

  for (int i = 0; i < in_h * in_w; i++)
    for (int j = 0; j < in_c_by_word; j++) {
      unsigned idx_in = nbits_per_word * (i * in_c_by_word + j);

      for (int bq = 0; bq < nbits_in_data; bq++) {
        T_q out[p::nbits_in_data];
        for (int i = 0; i < nbits_in_data; i++) { out[i] = 0; }

        for (int bw = 0; bw < nbits_per_word; bw++) {
          T_q in = in_data[idx_in + bw];
          in = (in >> bq) & 0x1;
          in = in << bw;
          out[bq] |= in;
        }

        in_data_packed[idx_out++] = out[bq];
      }
    }

  return idx_out;
}

unsigned pack_input_channel_wise_hp(T_in in_data[], T_q in_data_packed[], int in_h, int in_w, int in_c, int pad)
{
  namespace p = conv1x1_params;

  unsigned idx_in = 0;
  unsigned idx_out = 0;

  for (int ih = -pad; ih < in_h + pad; ih++)
    for (int iw = -pad; iw < in_w + pad; iw++)
      for (int ic = 0; ic < p::in_c; ic += p::nbits_per_word) {
        bool valid = (ih >= 0) && (ih < in_h) && (iw >= 0) && (iw < in_w);

        for (int bq = 0; bq < p::nbits_in_data; bq++) {
          T_q out[p::nbits_in_data];
          for (int i = 0; i < p::nbits_in_data; i++) { out[i] = 0; }

          for (int bw = 0; bw < p::nbits_per_word; bw++) {
            T_q in = 0;
            if (valid) {
              in = in_data[idx_in + bw];
            }

            in = (in >> bq) & 0x1;
            in = in << bw;
            out[bq] |= in;
          }

          in_data_packed[idx_out++] = out[bq];
        }
        if (valid) {
          idx_in += p::nbits_per_word;
        }
      }

  return idx_out;
}

template <class T>
bool compare_output(T out[], T expected[], const string name, unsigned out_h, unsigned out_w, unsigned out_c)
{
  unsigned idx = 0;

  for (unsigned oh = 0; oh < out_h; oh++)
    for (unsigned ow = 0; ow < out_w; ow++)
      for (unsigned oc = 0; oc < out_c; oc++) {
        if (out[idx] != expected[idx]) {
          cout << "[" << name << "]"
               << " test failed..." << endl;
          cout << "out_h: " << oh << "\n"
               << "out_w: " << ow << "\n"
               << "out_c: " << oc << "\n"
               << endl;
          cout << "expect: " << expected[idx] << ", " << name << ": " << out[idx] << endl;
          return false;
        }
        idx++;
      }

  cout << "[" << name << "] test success!!!" << endl;
  return true;
}

void kernel_transform_NHWC_to_NoCHWNi(T_q in_k[], T_q out_k[], unsigned kn, unsigned kh, unsigned kw, unsigned kc,
                                      unsigned kn_in)
{
  unsigned idx_src = 0;
  const unsigned kn_out = kn / kn_in;

  for (unsigned no = 0; no < kn_out; no++)
    for (unsigned ni = 0; ni < kn_in; ni++)
      for (unsigned h = 0; h < kh; h++)
        for (unsigned w = 0; w < kw; w++)
          for (unsigned c = 0; c < kc; c++) {
            unsigned idx_dst = no * (kc * kh * kw * kn_in);
            idx_dst += c * (kh * kw * kn_in);
            idx_dst += h * (kw * kn_in);
            idx_dst += w * (kn_in);
            idx_dst += ni;
            out_k[idx_dst] = in_k[idx_src++];
          }
}

template <class T>
void kernel_transform_NHWC_to_NoHWCNi(T src[], T dst[], unsigned kn, unsigned kh, unsigned kw, unsigned kc,
                                      unsigned kn_in)
{
  unsigned idx_src = 0;
  const unsigned kn_out = kn / kn_in;

  for (unsigned no = 0; no < kn_out; no++)
    for (unsigned ni = 0; ni < kn_in; ni++)
      for (unsigned h = 0; h < kh; h++)
        for (unsigned w = 0; w < kw; w++)
          for (unsigned c = 0; c < kc; c++) {
            unsigned idx_dst = no * (kh * kw * kc * kn_in);
            idx_dst += h * (kw * kc * kn_in);
            idx_dst += w * (kc * kn_in);
            idx_dst += c * (kn_in);
            idx_dst += ni;
            dst[idx_dst] = src[idx_src++];
          }
}

template <class T>
void kernel_transform_NHWC_to_HWNoCNi(T src[], T dst[], unsigned kn, unsigned kh, unsigned kw, unsigned kc,
                                      unsigned kn_in)
{
  unsigned idx_src = 0;
  const unsigned kn_out = kn / kn_in;

  for (unsigned no = 0; no < kn_out; no++)
    for (unsigned ni = 0; ni < kn_in; ni++)
      for (unsigned h = 0; h < kh; h++)
        for (unsigned w = 0; w < kw; w++)
          for (unsigned c = 0; c < kc; c++) {
            unsigned idx_dst = h * (kw * kn_out * kc * kn_in);
            idx_dst += w * (kn_out * kc * kn_in);
            idx_dst += no * (kc * kn_in);
            idx_dst += c * (kn_in);
            idx_dst += ni;
            dst[idx_dst] = src[idx_src++];
          }
}

void input_packed_transform_HWBC(T_q src[], T_q dst[], unsigned ih, unsigned iw, unsigned ic, unsigned ib)
{
  unsigned idx_src = 0;
  for (unsigned h = 0; h < ih; h++)
    for (unsigned w = 0; w < iw; w++)
      for (unsigned c = 0; c < ic; c++)
        for (unsigned b = 0; b < ib; b++) {
          unsigned idx_dst = h * (iw * ib * ic);
          idx_dst += w * (ib * ic);
          idx_dst += b * (ic);
          idx_dst += c;
          dst[idx_dst] = src[idx_src++];
        }
}

template <class T>
void kernel_transform_NHWC_to_HWCN(T src[], T dst[], unsigned kn, unsigned kh, unsigned kw, unsigned kc)
{
  unsigned idx_src = 0;

  for (unsigned n = 0; n < kn; n++)
    for (unsigned h = 0; h < kh; h++)
      for (unsigned w = 0; w < kw; w++)
        for (unsigned c = 0; c < kc; c++) {
          unsigned idx_dst = h * (kw * kc * kn);
          idx_dst += w * (kc * kn);
          idx_dst += c * (kn);
          idx_dst += n;
          dst[idx_dst] = src[idx_src++];
        }
}

template <class T>
void print_array(T array[], size_t n)
{
  for (size_t i = 0; i < n; ++i) { std::cout << "[" << i << "]: " << array[i] << std::endl; }
}

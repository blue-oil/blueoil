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

#include "global.h"
#include "matrix_view.h"
#include "operators.h" // FIXME(nikolay): for binary_convolution_parameters definition, rid of it later
#include "time_measurement.h"

#include <x86intrin.h>

namespace dlk {

namespace impl {

constexpr std::size_t MAX_CHANNELS = 4096;

static const auto buf_ts0 = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_CHANNELS);
static const auto buf_ts1 = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_CHANNELS);
static const auto buf_ts2 = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_CHANNELS);
static const auto buf_flg = std::make_unique<BIN_CONV_OUTPUT[]>(MAX_CHANNELS);


void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");

  for (unsigned int i = 0; i < result.rows(); ++i) {
    T_INT ts0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i];
    T_INT ts1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
    T_INT ts2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
    T_INT flag = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flag == -1) {
      ++ts0;
      ++ts1;
      ++ts2;
    }
    buf_ts0[i] = ts0;
    buf_ts1[i] = ts1;
    buf_ts2[i] = ts2;
    buf_flg[i] = flag;
  }

#pragma omp parallel for
  for (unsigned int j = 0; j < result.cols(); ++j) {
    for (unsigned int i = 0; i < result.rows(); i += 16) {
      const auto d = _mm256_loadu_si256(reinterpret_cast<__m256i*>(result.data(i, j)));
      const auto ts0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts0.get() + i));
      const auto ts1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts1.get() + i));
      const auto ts2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts2.get() + i));
      const auto flg = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_flg.get() + i));
      const auto f0 = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts0, d), flg);
      const auto f1 = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts1, d), flg);
      const auto f2 = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts2, d), flg);
      const auto is_neg = _mm256_cmpgt_epi16(_mm256_setzero_si256(), flg);
      const auto tmp = _mm256_add_epi16(_mm256_add_epi16(f0, f1), _mm256_add_epi16(f2, is_neg));
      const auto m2 = _mm256_sub_epi16(flg, _mm256_set1_epi16(2));
      const auto is_not_const = _mm256_cmpgt_epi16(_mm256_setzero_si256(), m2);
      const auto res = _mm256_blendv_epi8(m2, tmp, is_not_const);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data(i, j)), res);
    }
  }

  Measurement::Stop();
}

void ApplyThresholdsAndPack(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p,
    QUANTIZED_PACKED output[]) {
  Measurement::Start("ApplyThresholdsAndPack");

  for (unsigned int i = 0; i < result.rows(); ++i) {
    T_INT ts0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i];
    T_INT ts1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
    T_INT ts2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
    T_INT flag = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flag == -1) {
      ++ts0;
      ++ts1;
      ++ts2;
    }
    buf_ts0[i] = ts0;
    buf_ts1[i] = ts1;
    buf_ts2[i] = ts2;
    buf_flg[i] = flag;
  }

  const auto count = result.cols();
  const auto count_floor = count - (count % 4);

#pragma omp parallel for
  for (unsigned int j = 0; j < count_floor; j += 4) {
    for (unsigned int i = 0; i < result.rows(); i += 32) {
#define LOAD_TH(k) \
      const auto ts0##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts0.get() + i + k * 16)); \
      const auto ts1##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts1.get() + i + k * 16)); \
      const auto ts2##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts2.get() + i + k * 16)); \
      const auto flg##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_flg.get() + i + k * 16)); \
      const auto is_neg##k = _mm256_cmpgt_epi16(_mm256_setzero_si256(), flg##k); \
      const auto m2##k = _mm256_sub_epi16(flg##k, _mm256_set1_epi16(2)); \
      const auto is_not_const##k = _mm256_cmpgt_epi16(_mm256_setzero_si256(), m2##k);

#define APPLY(k, l) \
      const auto d##k##l = _mm256_loadu_si256(reinterpret_cast<__m256i*>(result.data(i + k * 16, j + l))); \
      const auto f0##k##l = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts0##k, d##k##l), flg##k); \
      const auto f1##k##l = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts1##k, d##k##l), flg##k); \
      const auto f2##k##l = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts2##k, d##k##l), flg##k); \
      const auto tmp##k##l = _mm256_add_epi16(_mm256_add_epi16(f0##k##l, f1##k##l), _mm256_add_epi16(f2##k##l, is_neg##k)); \
      const auto res##k##l = _mm256_blendv_epi8(m2##k, tmp##k##l, is_not_const##k);

#define PACK(l) \
      const auto packed##l = _mm256_packs_epi16(res0##l, res1##l); \
      const auto permuted##l = _mm256_permute4x64_epi64(packed##l, 0xD8); \
      const auto vlsb##l = _mm256_slli_epi32(permuted##l, 7); \
      const auto vmsb##l = _mm256_slli_epi32(permuted##l, 6); \
      const auto lsb##l = _mm256_movemask_epi8(vlsb##l); \
      const auto msb##l = _mm256_movemask_epi8(vmsb##l);

      LOAD_TH(0)
      LOAD_TH(1)
      APPLY(0, 0)
      APPLY(1, 0)
      APPLY(0, 1)
      APPLY(1, 1)
      APPLY(0, 2)
      APPLY(1, 2)
      APPLY(0, 3)
      APPLY(1, 3)
      PACK(0)
      PACK(1)
      PACK(2)
      PACK(3)
      const auto index = (j + (i / 32) * result.cols()) * 2;
      output[index+0] = QUANTIZED_PACKED(lsb0);
      output[index+1] = QUANTIZED_PACKED(msb0);
      output[index+2] = QUANTIZED_PACKED(lsb1);
      output[index+3] = QUANTIZED_PACKED(msb1);
      output[index+4] = QUANTIZED_PACKED(lsb2);
      output[index+5] = QUANTIZED_PACKED(msb2);
      output[index+6] = QUANTIZED_PACKED(lsb3);
      output[index+7] = QUANTIZED_PACKED(msb3);
#undef APPLY
    }
  }

  for (unsigned int j = count_floor; j < count; ++j) {
    for (unsigned int i = 0; i < result.rows(); i += 32) {
#define APPLY(k) \
      const auto d##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(result.data(i + k * 16, j))); \
      const auto ts0##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts0.get() + i + k * 16)); \
      const auto ts1##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts1.get() + i + k * 16)); \
      const auto ts2##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_ts2.get() + i + k * 16)); \
      const auto flg##k = _mm256_loadu_si256(reinterpret_cast<__m256i*>(buf_flg.get() + i + k * 16)); \
      const auto f0##k = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts0##k, d##k), flg##k); \
      const auto f1##k = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts1##k, d##k), flg##k); \
      const auto f2##k = _mm256_andnot_si256(_mm256_cmpgt_epi16(ts2##k, d##k), flg##k); \
      const auto is_neg##k = _mm256_cmpgt_epi16(_mm256_setzero_si256(), flg##k); \
      const auto tmp##k = _mm256_add_epi16(_mm256_add_epi16(f0##k, f1##k), _mm256_add_epi16(f2##k, is_neg##k)); \
      const auto m2##k = _mm256_sub_epi16(flg##k, _mm256_set1_epi16(2)); \
      const auto is_not_const##k = _mm256_cmpgt_epi16(_mm256_setzero_si256(), m2##k); \
      const auto res##k = _mm256_blendv_epi8(m2##k, tmp##k, is_not_const##k);
      APPLY(0)
      APPLY(1)
      const auto packed = _mm256_packs_epi16(res0, res1);
      const auto permuted = _mm256_permute4x64_epi64(packed, 0xD8);
      const auto vlsb = _mm256_slli_epi32(permuted, 7);
      const auto vmsb = _mm256_slli_epi32(permuted, 6);
      const auto lsb = _mm256_movemask_epi8(vlsb);
      const auto msb = _mm256_movemask_epi8(vmsb);
      const auto index = (j + (i / 32) * result.cols()) * 2;
      output[index] = QUANTIZED_PACKED(lsb);
      output[index+1] = QUANTIZED_PACKED(msb);
    }
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

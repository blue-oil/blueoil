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

#include <memory>
#include <arm_neon.h>

namespace dlk {

namespace impl {

void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");

  const auto th = std::make_unique<BIN_CONV_OUTPUT[]>(NUM_OF_A2W1_THRESHOLD * result.rows());
  for (unsigned int i = 0; i < result.rows(); ++i) {
    auto th0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 0];
    auto th1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
    auto th2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
    const auto flg = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flg == -1) {
      ++th0;
      ++th1;
      ++th2;
    }
    th[NUM_OF_A2W1_THRESHOLD * i + 0] = th0;
    th[NUM_OF_A2W1_THRESHOLD * i + 1] = th1;
    th[NUM_OF_A2W1_THRESHOLD * i + 2] = th2;
    th[NUM_OF_A2W1_THRESHOLD * i + 3] = flg;
  }

#pragma omp parallel for
  for (unsigned int j = 0; j < result.cols(); ++j) {
    for (unsigned int i = 0; i < result.rows(); i += 8) {
      const auto d = vld1q_s16(result.data(i, j));
      const auto ts = vld4q_s16(th.get() + NUM_OF_A2W1_THRESHOLD * i);
      const auto f0 = vreinterpretq_s16_u16(vcgeq_s16(d, ts.val[0])) & ts.val[3];
      const auto f1 = vreinterpretq_s16_u16(vcgeq_s16(d, ts.val[1])) & ts.val[3];
      const auto f2 = vreinterpretq_s16_u16(vcgeq_s16(d, ts.val[2])) & ts.val[3];
      const auto is_neg = vreinterpretq_s16_u16(vcltq_s16(ts.val[3], vdupq_n_s16(0)));
      const auto tmp = f0 + f1 + f2 + is_neg;
      const auto m2 = vsubq_s16(ts.val[3], vdupq_n_s16(2));
      const auto is_const = vcgeq_s16(m2, vdupq_n_s16(0));
      const auto res = vbslq_s16(is_const, m2, tmp);
      vst1q_s16(result.data(i, j), res);
    }
  }

  Measurement::Stop();
}

void ApplyThresholdsAndPack(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p,
    QUANTIZED_PACKED output[]) {
  Measurement::Start("ApplyThresholdsAndPack");

  const auto th = std::make_unique<BIN_CONV_OUTPUT[]>(NUM_OF_A2W1_THRESHOLD * result.rows());
  for (unsigned int i = 0; i < result.rows(); ++i) {
    auto th0 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 0];
    auto th1 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 1];
    auto th2 = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 2];
    const auto flg = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + 3];
    if (flg == -1) {
      ++th0;
      ++th1;
      ++th2;
    }
    th[NUM_OF_A2W1_THRESHOLD * i + 0] = th0;
    th[NUM_OF_A2W1_THRESHOLD * i + 1] = th1;
    th[NUM_OF_A2W1_THRESHOLD * i + 2] = th2;
    th[NUM_OF_A2W1_THRESHOLD * i + 3] = flg;
  }

#define APPLY(k) \
  const auto col##k = (i + 8 * k) / result.rows(); \
  const auto row##k = (i + 8 * k) % result.rows(); \
  const auto d##k = vld1q_s16(result.data(row##k, col##k)); \
  const auto ts##k = vld4q_s16(th.get() + NUM_OF_A2W1_THRESHOLD * row##k); \
  const auto f##k##0 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[0])) & ts##k.val[3]; \
  const auto f##k##1 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[1])) & ts##k.val[3]; \
  const auto f##k##2 = vreinterpretq_s16_u16(vcgeq_s16(d##k, ts##k.val[2])) & ts##k.val[3]; \
  const auto is_neg##k = vreinterpretq_s16_u16(vcltq_s16(ts##k.val[3], vdupq_n_s16(0))); \
  const auto tmp##k = f##k##0 + f##k##1 + f##k##2 + is_neg##k; \
  const auto m2_##k = vsubq_s16(ts##k.val[3], vdupq_n_s16(2)); \
  const auto is_const##k = vcgeq_s16(m2_##k, vdupq_n_s16(0)); \
  const auto res##k = vreinterpretq_u8_s16(vbslq_s16(is_const##k, m2_##k, tmp##k));

#define MAKE_B(k, k0, k1) \
  const auto a##k = vuzpq_u8(res##k0, res##k1).val[0]; \
  const auto am##k = vmulq_u8(vshrq_n_u8(a##k, 1), coeff); \
  const auto al##k = vmulq_u8(vandq_u8(a##k, vdupq_n_u8(0x01)), coeff); \
  const auto bm##k = vreinterpretq_u8_u16(vpaddlq_u8(am##k)); \
  const auto bl##k = vreinterpretq_u8_u16(vpaddlq_u8(al##k));

#define MAKE_D(k, k0, k1) \
  const auto cm##k = vuzpq_u8(bm##k0, bm##k1).val[0]; \
  const auto cl##k = vuzpq_u8(bl##k0, bl##k1).val[0]; \
  const auto dm##k = vreinterpretq_u8_u16(vpaddlq_u8(cm##k)); \
  const auto dl##k = vreinterpretq_u8_u16(vpaddlq_u8(cl##k));

#define MAKE_F(k, k0, k1) \
  const auto em##k = vuzpq_u8(dm##k0, dm##k1).val[0]; \
  const auto el##k = vuzpq_u8(dl##k0, dl##k1).val[0]; \
  const auto fm##k = vreinterpretq_u8_u16(vpaddlq_u8(em##k)); \
  const auto fl##k = vreinterpretq_u8_u16(vpaddlq_u8(el##k));

  const auto length = result.cols() * result.rows();
  constexpr uint8_t coeff_ary[16] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };
  const auto coeff = vld1q_u8(coeff_ary);
#ifdef AARCH32
  constexpr std::size_t SIMD_WIDTH = 64; // hard coded, not configurable
#pragma omp parallel for
  for (std::size_t i = 0; i < length; i += 32) {
    APPLY(0)
    APPLY(1)
    MAKE_B(0, 0, 1)
    APPLY(2)
    APPLY(3)
    MAKE_B(1, 2, 3)
    // store
    const auto cm = vuzpq_u8(bm0, bm1).val[0];
    const auto cl = vuzpq_u8(bl0, bl1).val[0];
    const auto dm = vreinterpret_u8_u16(vmovn_u32(vpaddlq_u16(vpaddlq_u8(cm))));
    const auto dl = vreinterpret_u8_u16(vmovn_u32(vpaddlq_u16(vpaddlq_u8(cl))));
    const auto e = vreinterpret_u32_u8(vuzp_u8(dl, dm).val[0]);
    vst1_u32(reinterpret_cast<uint32_t*>(output + i / 16), e);
  }
#else
  constexpr std::size_t SIMD_WIDTH = 128; // hard coded, not configurable
  const auto length_floor = length - (length % SIMD_WIDTH);
#pragma omp parallel for
  for (std::size_t i = 0; i < length_floor; i += SIMD_WIDTH) {
    APPLY(0)
    APPLY(1)
    MAKE_B(0, 0, 1)
    APPLY(2)
    APPLY(3)
    MAKE_B(1, 2, 3)
    MAKE_D(0, 0, 1)
    APPLY(4)
    APPLY(5)
    MAKE_B(2, 4, 5)
    APPLY(6)
    APPLY(7)
    MAKE_B(3, 6, 7)
    MAKE_D(1, 2, 3)
    MAKE_F(0, 0, 1)
    APPLY(8)
    APPLY(9)
    MAKE_B(4, 8, 9)
    APPLY(10)
    APPLY(11)
    MAKE_B(5, 10, 11)
    MAKE_D(2, 4, 5)
    APPLY(12)
    APPLY(13)
    MAKE_B(6, 12, 13)
    APPLY(14)
    APPLY(15)
    MAKE_B(7, 14, 15)
    MAKE_D(3, 6, 7)
    MAKE_F(1, 2, 3)
    // g
    uint32x4x2_t g;
    g.val[1] = vreinterpretq_u32_u8(vuzpq_u8(fm0, fm1).val[0]);
    g.val[0] = vreinterpretq_u32_u8(vuzpq_u8(fl0, fl1).val[0]);
    vst2q_u32(reinterpret_cast<uint32_t*>(output + i / 16), g);
  }
  for (std::size_t i = length_floor; i < length; i += 32) {
    APPLY(0)
    APPLY(1)
    MAKE_B(0, 0, 1)
    APPLY(2)
    APPLY(3)
    MAKE_B(1, 2, 3)
    // store
    const auto cm = vuzpq_u8(bm0, bm1).val[0];
    const auto cl = vuzpq_u8(bl0, bl1).val[0];
    const auto dm = vreinterpret_u8_u16(vmovn_u32(vpaddlq_u16(vpaddlq_u8(cm))));
    const auto dl = vreinterpret_u8_u16(vmovn_u32(vpaddlq_u16(vpaddlq_u8(cl))));
    const auto e = vreinterpret_u32_u8(vuzp_u8(dl, dm).val[0]);
    vst1_u32(reinterpret_cast<uint32_t*>(output + i / 16), e);
  }
#endif
#undef APPLY
#undef MAKE_B
#undef MAKE_D
#undef MAKE_F

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

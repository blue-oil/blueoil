/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

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

#include "func/impl/pack_16bit.h"
#include <cassert>

#include <arm_neon.h>

#include "time_measurement.h"

namespace dlk {

namespace impl {

void pack_16bit(const BIN_CONV_OUTPUT input[], QUANTIZED_PACKED output[], const std::size_t length) {
  using base = QUANTIZED_PACKED::base_t;
  const auto bits = QUANTIZED_PACKED::BitCount;
  assert((length % bits) == 0);
  Measurement::Start("pack bits");
  std::size_t i = 0, j = 0;
#ifdef AARCH32
  constexpr std::size_t SIMD_WIDTH = 64; // hardcoded, not configurable
  constexpr uint8_t coeff_ary[16] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };
  const auto coeff = vld1q_u8(coeff_ary);
  for (; i < length; i += SIMD_WIDTH) {
    // b0
    const auto a0 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x00)).val[0];
    const auto am0 = vmulq_u8(vshrq_n_u8(a0, 1), coeff);
    const auto al0 = vmulq_u8(vandq_u8(a0, vdupq_n_u8(0x01)), coeff);
    const auto bm0 = vreinterpretq_u8_u16(vpaddlq_u8(am0));
    const auto bl0 = vreinterpretq_u8_u16(vpaddlq_u8(al0));
    // b1
    const auto a1 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x10)).val[0];
    const auto am1 = vmulq_u8(vshrq_n_u8(a1, 1), coeff);
    const auto al1 = vmulq_u8(vandq_u8(a1, vdupq_n_u8(0x01)), coeff);
    const auto bm1 = vreinterpretq_u8_u16(vpaddlq_u8(am1));
    const auto bl1 = vreinterpretq_u8_u16(vpaddlq_u8(al1));
    // d01
    const auto cm01 = vuzpq_u8(bm0, bm1).val[0];
    const auto cl01 = vuzpq_u8(bl0, bl1).val[0];
    const auto dm01 = vreinterpretq_u8_u16(vpaddlq_u8(cm01));
    const auto dl01 = vreinterpretq_u8_u16(vpaddlq_u8(cl01));
    // b2
    const auto a2 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x20)).val[0];
    const auto am2 = vmulq_u8(vshrq_n_u8(a2, 1), coeff);
    const auto al2 = vmulq_u8(vandq_u8(a2, vdupq_n_u8(0x01)), coeff);
    const auto bm2 = vreinterpretq_u8_u16(vpaddlq_u8(am2));
    const auto bl2 = vreinterpretq_u8_u16(vpaddlq_u8(al2));
    // b3
    const auto a3 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x30)).val[0];
    const auto am3 = vmulq_u8(vshrq_n_u8(a3, 1), coeff);
    const auto al3 = vmulq_u8(vandq_u8(a3, vdupq_n_u8(0x01)), coeff);
    const auto bm3 = vreinterpretq_u8_u16(vpaddlq_u8(am3));
    const auto bl3 = vreinterpretq_u8_u16(vpaddlq_u8(al3));
    // d23
    const auto cm23 = vuzpq_u8(bm2, bm3).val[0];
    const auto cl23 = vuzpq_u8(bl2, bl3).val[0];
    const auto dm23 = vreinterpretq_u8_u16(vpaddlq_u8(cm23));
    const auto dl23 = vreinterpretq_u8_u16(vpaddlq_u8(cl23));
    // g
    const auto em = vuzpq_u8(dm01, dm23).val[0];
    const auto el = vuzpq_u8(dl01, dl23).val[0];
    const auto fm = vreinterpret_u32_u8(vmovn_u16(vpaddlq_u8(em)));
    const auto fl = vreinterpret_u32_u8(vmovn_u16(vpaddlq_u8(el)));
    const auto g = vzip_u32(fl, fm);
    vst1_u32(reinterpret_cast<uint32_t*>(output + j), g.val[0]);
    vst1_u32(reinterpret_cast<uint32_t*>(output + j + 2), g.val[1]);
    j += 4;
  }
#else
  constexpr std::size_t SIMD_WIDTH = 128; // hardcoded, not configurable
  constexpr uint8_t coeff_ary[16] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };
  const auto coeff = vld1q_u8(coeff_ary);
  for (; i < length; i += SIMD_WIDTH) {
    // b0
    const auto a0 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x00)).val[0];
    const auto am0 = vmulq_u8(vshrq_n_u8(a0, 1), coeff);
    const auto al0 = vmulq_u8(vandq_u8(a0, vdupq_n_u8(0x01)), coeff);
    const auto bm0 = vreinterpretq_u8_u16(vpaddlq_u8(am0));
    const auto bl0 = vreinterpretq_u8_u16(vpaddlq_u8(al0));
    // b1
    const auto a1 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x10)).val[0];
    const auto am1 = vmulq_u8(vshrq_n_u8(a1, 1), coeff);
    const auto al1 = vmulq_u8(vandq_u8(a1, vdupq_n_u8(0x01)), coeff);
    const auto bm1 = vreinterpretq_u8_u16(vpaddlq_u8(am1));
    const auto bl1 = vreinterpretq_u8_u16(vpaddlq_u8(al1));
    // d01
    const auto cm01 = vuzpq_u8(bm0, bm1).val[0];
    const auto cl01 = vuzpq_u8(bl0, bl1).val[0];
    const auto dm01 = vreinterpretq_u8_u16(vpaddlq_u8(cm01));
    const auto dl01 = vreinterpretq_u8_u16(vpaddlq_u8(cl01));
    // b2
    const auto a2 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x20)).val[0];
    const auto am2 = vmulq_u8(vshrq_n_u8(a2, 1), coeff);
    const auto al2 = vmulq_u8(vandq_u8(a2, vdupq_n_u8(0x01)), coeff);
    const auto bm2 = vreinterpretq_u8_u16(vpaddlq_u8(am2));
    const auto bl2 = vreinterpretq_u8_u16(vpaddlq_u8(al2));
    // b3
    const auto a3 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x30)).val[0];
    const auto am3 = vmulq_u8(vshrq_n_u8(a3, 1), coeff);
    const auto al3 = vmulq_u8(vandq_u8(a3, vdupq_n_u8(0x01)), coeff);
    const auto bm3 = vreinterpretq_u8_u16(vpaddlq_u8(am3));
    const auto bl3 = vreinterpretq_u8_u16(vpaddlq_u8(al3));
    // d23
    const auto cm23 = vuzpq_u8(bm2, bm3).val[0];
    const auto cl23 = vuzpq_u8(bl2, bl3).val[0];
    const auto dm23 = vreinterpretq_u8_u16(vpaddlq_u8(cm23));
    const auto dl23 = vreinterpretq_u8_u16(vpaddlq_u8(cl23));
    // f0123
    const auto em0123 = vuzpq_u8(dm01, dm23).val[0];
    const auto el0123 = vuzpq_u8(dl01, dl23).val[0];
    const auto fm0123 = vreinterpretq_u8_u16(vpaddlq_u8(em0123));
    const auto fl0123 = vreinterpretq_u8_u16(vpaddlq_u8(el0123));
    // b4
    const auto a4 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x40)).val[0];
    const auto am4 = vmulq_u8(vshrq_n_u8(a4, 1), coeff);
    const auto al4 = vmulq_u8(vandq_u8(a4, vdupq_n_u8(0x01)), coeff);
    const auto bm4 = vreinterpretq_u8_u16(vpaddlq_u8(am4));
    const auto bl4 = vreinterpretq_u8_u16(vpaddlq_u8(al4));
    // b5
    const auto a5 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x50)).val[0];
    const auto am5 = vmulq_u8(vshrq_n_u8(a5, 1), coeff);
    const auto al5 = vmulq_u8(vandq_u8(a5, vdupq_n_u8(0x01)), coeff);
    const auto bm5 = vreinterpretq_u8_u16(vpaddlq_u8(am5));
    const auto bl5 = vreinterpretq_u8_u16(vpaddlq_u8(al5));
    // d45
    const auto cm45 = vuzpq_u8(bm4, bm5).val[0];
    const auto cl45 = vuzpq_u8(bl4, bl5).val[0];
    const auto dm45 = vreinterpretq_u8_u16(vpaddlq_u8(cm45));
    const auto dl45 = vreinterpretq_u8_u16(vpaddlq_u8(cl45));
    // b6
    const auto a6 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x60)).val[0];
    const auto am6 = vmulq_u8(vshrq_n_u8(a6, 1), coeff);
    const auto al6 = vmulq_u8(vandq_u8(a6, vdupq_n_u8(0x01)), coeff);
    const auto bm6 = vreinterpretq_u8_u16(vpaddlq_u8(am6));
    const auto bl6 = vreinterpretq_u8_u16(vpaddlq_u8(al6));
    // b7
    const auto a7 = vld2q_u8(reinterpret_cast<const uint8_t*>(input + i + 0x70)).val[0];
    const auto am7 = vmulq_u8(vshrq_n_u8(a7, 1), coeff);
    const auto al7 = vmulq_u8(vandq_u8(a7, vdupq_n_u8(0x01)), coeff);
    const auto bm7 = vreinterpretq_u8_u16(vpaddlq_u8(am7));
    const auto bl7 = vreinterpretq_u8_u16(vpaddlq_u8(al7));
    // d67
    const auto cm67 = vuzpq_u8(bm6, bm7).val[0];
    const auto cl67 = vuzpq_u8(bl6, bl7).val[0];
    const auto dm67 = vreinterpretq_u8_u16(vpaddlq_u8(cm67));
    const auto dl67 = vreinterpretq_u8_u16(vpaddlq_u8(cl67));
    // f4567
    const auto em4567 = vuzpq_u8(dm45, dm67).val[0];
    const auto el4567 = vuzpq_u8(dl45, dl67).val[0];
    const auto fm4567 = vreinterpretq_u8_u16(vpaddlq_u8(em4567));
    const auto fl4567 = vreinterpretq_u8_u16(vpaddlq_u8(el4567));
    // g
    uint32x4x2_t g;
    g.val[1] = vreinterpretq_u32_u8(vuzpq_u8(fm0123, fm4567).val[0]);
    g.val[0] = vreinterpretq_u32_u8(vuzpq_u8(fl0123, fl4567).val[0]);
    vst2q_u32(reinterpret_cast<uint32_t*>(output + j), g);
    j += 8;
  }
#endif
  QUANTIZED_PACKED lsb(0), msb(0);
  for (; i < length; i += bits) {
    for (std::size_t i2 = 0; i2 < bits; ++i2) {
      msb |= QUANTIZED_PACKED((base)((input[i+i2] >> 1) & 1) << i2);
      lsb |= QUANTIZED_PACKED((base)(input[i+i2] & 1) << i2);
    }
    output[j] = lsb;
    output[j+1] = msb;
    lsb = QUANTIZED_PACKED(0);
    msb = QUANTIZED_PACKED(0);
    j += 2;
  }
  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

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

#include <x86intrin.h>

#include "time_measurement.h"

namespace dlk {

namespace impl {

void pack_16bit(const BIN_CONV_OUTPUT input[], QUANTIZED_PACKED output[], const std::size_t length) {
  using base = QUANTIZED_PACKED::base_t;
  const auto bits = QUANTIZED_PACKED::BitCount;
  assert((length % bits) == 0);
  Measurement::Start("pack bits");
  std::size_t j = 0;
  for (std::size_t i = 0; i < length; i += bits) {
    const auto v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
    const auto v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i + 16));
    const auto packed = _mm256_packs_epi16(v1, v2);
    const auto permuted = _mm256_permute4x64_epi64(packed, 0xD8);
    const auto vlsb = _mm256_slli_epi32(permuted, 7);
    const auto vmsb = _mm256_slli_epi32(permuted, 6);
    const auto lsb = _mm256_movemask_epi8(vlsb);
    const auto msb = _mm256_movemask_epi8(vmsb);
    output[j] = QUANTIZED_PACKED(lsb);
    output[j+1] = QUANTIZED_PACKED(msb);
    j += 2;
  }
  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

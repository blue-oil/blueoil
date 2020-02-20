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

#include "time_measurement.h"

namespace dlk {

namespace impl {

void pack_16bit(const BIN_CONV_OUTPUT input[], QUANTIZED_PACKED output[], const std::size_t length) {
  using base = QUANTIZED_PACKED::base_t;
  const auto bits = QUANTIZED_PACKED::BitCount;
  assert((length % bits) == 0);
  Measurement::Start("pack bits");
  std::size_t j = 0;
  QUANTIZED_PACKED msb(0), lsb(0);
  for (std::size_t i = 0; i < length; i += bits) {
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

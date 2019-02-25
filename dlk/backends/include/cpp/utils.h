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

namespace cpp {

inline T_out pop_count(T_in in_data)
{
  T_out sum = 0;
  const unsigned nbits_per_word = sizeof(T_in) * 8;
  for (unsigned i = 0; i < nbits_per_word; i++) { sum += (in_data >> i) & 0x1; }
  return sum;
}

inline T_out PE(T_q k_buf, T_q in_buf0, T_q in_buf1)
{
  T_q xnor0 = ~(in_buf0 ^ k_buf);
  T_q xnor1 = ~(in_buf1 ^ k_buf);
  T_out in_pc0 = pop_count(xnor0);
  T_out in_pc1 = pop_count(xnor1);
  T_out k_pc = pop_count(~k_buf);

  return in_pc0 + (2 * in_pc1) - (3 * k_pc);
}

} // namespace cpp

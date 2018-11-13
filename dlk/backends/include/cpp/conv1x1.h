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
void conv1x1_impl(T_in in_data[], T_out out_data[], T_k k_data[], unsigned in_w, unsigned in_h, unsigned in_c,
                  unsigned out_c);

void qconv1x1_impl(T_q in_data_packed[], T_out out_data[], T_q k_data_packed[], unsigned in_w, unsigned in_h,
                   unsigned in_c_by_word, unsigned out_c);

} // namespace cpp

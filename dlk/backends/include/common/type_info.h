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
#include <inttypes.h>

namespace type {
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
} // namespace type

typedef uint32_t T_q;
typedef T_q T_in;
typedef int16_t T_out;
typedef int32_t T_k;

#if defined __arm__
typedef uint8_t T_in_k2c;
typedef int16_t T_out_k2c;
typedef int8_t T_k_k2c;
#else
typedef unsigned char T_in_k2c;
typedef short T_out_k2c;
typedef char T_k_k2c;
#endif

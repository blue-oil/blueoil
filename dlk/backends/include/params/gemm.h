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

typedef struct Gemm_params_type
{
  const unsigned num_pe;
  const unsigned nbits_per_word;
  const unsigned nbits_in_data;
  const unsigned nbits_w_data;

  const unsigned max_in_h;
  const unsigned max_in_w;
  const unsigned max_in_w_by_word;
  const unsigned max_w_h;
  const unsigned max_w_h_by_word;

  const unsigned in_h;
  const unsigned in_w;
  const unsigned in_w_by_word;
  const unsigned in_size;
  const unsigned in_size_packed;

  const unsigned w_h;
  const unsigned w_w;
  const unsigned w_h_by_word;
  const unsigned w_size;
  const unsigned w_size_packed;

  const unsigned out_h;
  const unsigned out_w;
  const unsigned out_size;
} Gemm_params_t;

#define new_Gemm_params(NAME) \
  {NAME::num_pe,              \
   NAME::nbits_per_word,      \
   NAME::nbits_in_data,       \
   NAME::nbits_w_data,        \
                              \
   NAME::max_in_h,            \
   NAME::max_in_w,            \
   NAME::max_in_w_by_word,    \
   NAME::max_w_h,             \
   NAME::max_w_h_by_word,     \
                              \
   NAME::in_h,                \
   NAME::in_w,                \
   NAME::in_w_by_word,        \
   NAME::in_size,             \
   NAME::in_size_packed,      \
                              \
   NAME::w_h,                 \
   NAME::w_w,                 \
   NAME::w_h_by_word,         \
   NAME::w_size,              \
   NAME::w_size_packed,       \
                              \
   NAME::out_h,               \
   NAME::out_w,               \
   NAME::out_size};

namespace gemm_params {
static const unsigned num_pe = 8;
static const unsigned nbits_per_word = 32;
static const unsigned nbits_in_data = 2;
static const unsigned nbits_w_data = 1;

static const unsigned max_in_h = 64;
static const unsigned max_in_w = 64;
static const unsigned max_in_w_by_word = max_in_w / nbits_per_word;
static const unsigned max_w_h = max_in_h;
static const unsigned max_w_h_by_word = max_w_h / nbits_per_word;

static const unsigned in_h = 4;
static const unsigned in_w = 4;
static const unsigned in_w_by_word = max_in_w / nbits_per_word;
static const unsigned in_size = in_h * in_w;
static const unsigned in_size_packed = in_h * in_w_by_word * nbits_in_data;

static const unsigned w_h = 4;
static const unsigned w_w = 4;
static const unsigned w_h_by_word = w_h / nbits_per_word;
static const unsigned w_size = w_h * w_w;
static const unsigned w_size_packed = w_h * w_h_by_word & nbits_w_data;

static const unsigned out_h = in_h;
static const unsigned out_w = w_w;
static const unsigned out_size = out_h * out_w;
} // namespace gemm_params

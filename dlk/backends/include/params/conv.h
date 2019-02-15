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

typedef struct Conv_params_type
{
  const unsigned num_pe;
  const unsigned nbits_per_word;
  const unsigned nbits_in_data;
  const unsigned nbits_k_data;
  const unsigned num_thresholds;

  const unsigned pad_w;
  const unsigned pad_h;
  const unsigned stride_w;
  const unsigned stride_h;

  const unsigned max_in_w;
  const unsigned max_in_h;
  const unsigned max_in_w_with_pad;
  const unsigned max_in_h_with_pad;
  const unsigned max_in_c;
  const unsigned max_in_c_by_word;
  const unsigned min_in_c;
  const unsigned min_in_c_by_word;
  const unsigned max_k_c;
  const unsigned max_k_c_by_word;

  const unsigned num_in_by_unit;
  const unsigned log_num_in_by_unit;
  const unsigned max_num_in_unit;

  const unsigned in_w;
  const unsigned in_h;
  const unsigned in_w_with_pad;
  const unsigned in_h_with_pad;
  const unsigned in_c;
  const unsigned in_c_by_word;
  const unsigned in_size;
  const unsigned in_size_packed;

  const unsigned k_h;
  const unsigned k_w;
  const unsigned k_c;
  const unsigned k_n;
  const unsigned k_c_by_word;
  const unsigned k_size;

  const unsigned out_h;
  const unsigned out_w;
  const unsigned out_c;
  const unsigned out_size;

  const unsigned in_size_hp;
  const unsigned in_size_hp_packed;
  const unsigned in_size_hp_packed_spec;
  const unsigned k_size_packed;

  const unsigned inb_h;
  const unsigned inb_w;
  const unsigned num_in_line;

  bool has_thresholds;
} Conv_params_t;

#define new_Conv_params(NAME)    \
  {NAME::num_pe,                 \
   NAME::nbits_per_word,         \
   NAME::nbits_in_data,          \
   NAME::nbits_k_data,           \
   NAME::num_thresholds,         \
                                 \
   NAME::pad_w,                  \
   NAME::pad_h,                  \
   NAME::stride_w,               \
   NAME::stride_h,               \
                                 \
   NAME::max_in_w,               \
   NAME::max_in_h,               \
   NAME::max_in_w_with_pad,      \
   NAME::max_in_h_with_pad,      \
   NAME::max_in_c,               \
   NAME::max_in_c_by_word,       \
   NAME::min_in_c,               \
   NAME::min_in_c_by_word,       \
   NAME::max_k_c,                \
   NAME::max_k_c_by_word,        \
                                 \
   NAME::num_in_by_unit,         \
   NAME::log_num_in_by_unit,     \
   NAME::max_num_in_unit,        \
                                 \
   NAME::in_w,                   \
   NAME::in_h,                   \
   NAME::in_w_with_pad,          \
   NAME::in_h_with_pad,          \
   NAME::in_c,                   \
   NAME::in_c_by_word,           \
   NAME::in_size,                \
   NAME::in_size_packed,         \
                                 \
   NAME::k_h,                    \
   NAME::k_w,                    \
   NAME::k_c,                    \
   NAME::k_n,                    \
   NAME::k_c_by_word,            \
   NAME::k_size,                 \
                                 \
   NAME::out_h,                  \
   NAME::out_w,                  \
   NAME::out_c,                  \
   NAME::out_size,               \
                                 \
   NAME::in_size_hp,             \
   NAME::in_size_hp_packed,      \
   NAME::in_size_hp_packed_spec, \
   NAME::k_size_packed,          \
                                 \
   NAME::inb_h,                  \
   NAME::inb_w,                  \
   NAME::num_in_line,            \
   NAME::has_thresholds};

namespace conv_common_params {
static const unsigned num_pe = 16;
static const unsigned nbits_per_word = 32;
static const unsigned nbits_in_data = 2;
static const unsigned nbits_k_data = 1;
static const unsigned num_thresholds = 4;

static const unsigned max_in_c = 1024;
static const unsigned max_in_c_by_word = max_in_c / nbits_per_word;
static const unsigned min_in_c = 32;
static const unsigned min_in_c_by_word = min_in_c / nbits_per_word;
static const unsigned max_in_b = 2;
static const unsigned min_in_b = 2;
} // namespace conv_common_params

namespace conv3x3_params {
static const unsigned num_pe = conv_common_params::num_pe;
static const unsigned nbits_per_word = conv_common_params::nbits_per_word;
static const unsigned nbits_in_data = conv_common_params::nbits_in_data;
static const unsigned nbits_k_data = conv_common_params::nbits_k_data;
static const unsigned num_thresholds = conv_common_params::num_thresholds;

static const unsigned pad_w = 1;
static const unsigned pad_h = 1;
static const unsigned stride_w = 1;
static const unsigned stride_h = 1;

static const unsigned max_in_w = 512;
static const unsigned max_in_h = 512;
static const unsigned max_in_w_with_pad = max_in_w + (2 * pad_w);
static const unsigned max_in_h_with_pad = max_in_h + (2 * pad_h);
static const unsigned max_in_c = conv_common_params::max_in_c;
static const unsigned max_in_c_by_word = conv_common_params::max_in_c_by_word;
static const unsigned min_in_c = conv_common_params::min_in_c;
static const unsigned min_in_c_by_word = conv_common_params::min_in_c_by_word;
static const unsigned max_k_c = max_in_c;
static const unsigned max_k_c_by_word = max_k_c / nbits_per_word;

static const unsigned num_in_by_unit = min_in_c_by_word; // 4
static const unsigned log_num_in_by_unit = 2;            // log2(num_in_by_unit) = log2(4) = 2
static const unsigned max_num_in_unit = (max_in_c_by_word + (num_in_by_unit - 1)) >> log_num_in_by_unit; // / 4

static const unsigned in_w = 32;
static const unsigned in_h = 32;
static const unsigned in_w_with_pad = in_w + (2 * pad_w);
static const unsigned in_h_with_pad = in_h + (2 * pad_h);
static const unsigned in_c = 128;
static const unsigned in_c_by_word = in_c / nbits_per_word;
static const unsigned in_size = in_h * in_w * in_c;
static const unsigned in_size_packed = in_h * in_w * in_c_by_word * nbits_in_data;

static const unsigned k_h = 3;
static const unsigned k_w = 3;
static const unsigned k_c = in_c;
static const unsigned k_n = num_pe * 2;
static const unsigned k_c_by_word = k_c / nbits_per_word;
static const unsigned k_size = k_h * k_w * k_c;

static const unsigned out_h = (in_h_with_pad - k_h) + 1;
static const unsigned out_w = (in_w_with_pad - k_w) + 1;
static const unsigned out_c = k_n;
static const unsigned out_size = out_h * out_w * out_c;

static const unsigned in_size_hp = (in_h + 2 * pad_h) * (in_w + 2 * pad_w) * in_c;
static const unsigned in_size_hp_packed = (in_h + 2 * pad_h) * (in_w + 2 * pad_w) * in_c_by_word * nbits_in_data;
static const unsigned in_size_hp_packed_spec =
  ((in_h + 1) + 2 * pad_h) * (in_w + 2 * pad_w) * in_c_by_word * nbits_in_data;
static const unsigned k_size_packed = k_h * k_w * k_c_by_word * nbits_k_data;

static const unsigned inb_h = k_h + 1;
static const unsigned inb_w = k_w + 1;
static const unsigned num_in_line = (max_in_w_with_pad * (k_h - 1)) + k_w;
static const bool has_thresholds = true;
} // namespace conv3x3_params

namespace conv1x1_params {
static const unsigned num_pe = conv3x3_params::num_pe;
static const unsigned nbits_per_word = conv3x3_params::nbits_per_word;
static const unsigned nbits_in_data = conv3x3_params::nbits_in_data;
static const unsigned nbits_k_data = conv3x3_params::nbits_k_data;
static const unsigned num_thresholds = conv_common_params::num_thresholds;

static const unsigned pad_w = 0;
static const unsigned pad_h = 0;
static const unsigned stride_w = 1;
static const unsigned stride_h = 1;

static const unsigned max_in_w = 64;
static const unsigned max_in_h = 64;
static const unsigned max_in_w_with_pad = max_in_w + (2 * pad_w);
static const unsigned max_in_h_with_pad = max_in_h + (2 * pad_h);
static const unsigned max_in_c = conv_common_params::max_in_c;
static const unsigned max_in_c_by_word = conv_common_params::max_in_c_by_word;
static const unsigned min_in_c = conv_common_params::min_in_c;
static const unsigned min_in_c_by_word = conv_common_params::min_in_c_by_word;
static const unsigned max_k_c = max_in_c;
static const unsigned max_k_c_by_word = max_k_c / nbits_per_word;

static const unsigned num_in_by_unit = min_in_c_by_word; // 4
static const unsigned log_num_in_by_unit = 2;            // log2(num_in_by_unit) = log2(4) = 2
static const unsigned max_num_in_unit = (max_in_c_by_word + (num_in_by_unit - 1)) >> log_num_in_by_unit; // / 4

static const unsigned in_w = 32;
static const unsigned in_h = 32;
static const unsigned in_w_with_pad = in_w + (2 * pad_w);
static const unsigned in_h_with_pad = in_h + (2 * pad_h);
static const unsigned in_c = 128;
static const unsigned in_c_by_word = in_c / nbits_per_word;
static const unsigned in_size = in_h * in_w * in_c;
static const unsigned in_size_packed = in_h * in_w * in_c_by_word * nbits_in_data;

static const unsigned k_h = 1;
static const unsigned k_w = 1;
static const unsigned k_c = in_c;
static const unsigned k_n = num_pe * 2;
static const unsigned k_c_by_word = k_c / nbits_per_word;
static const unsigned k_size = k_h * k_w * k_c;

static const unsigned out_h = (in_h_with_pad - k_h) + 1;
static const unsigned out_w = (in_w_with_pad - k_w) + 1;
static const unsigned out_c = k_n;
static const unsigned out_size = out_h * out_w * out_c;

static const unsigned in_size_hp = (in_h + 2 * pad_h) * (in_w + 2 * pad_w) * in_c;
static const unsigned in_size_hp_packed = (in_h + 2 * pad_h) * (in_w + 2 * pad_w) * in_c_by_word * nbits_in_data;
static const unsigned in_size_hp_packed_spec =
  ((in_h + 1) + 2 * pad_h) * (in_w + 2 * pad_w) * in_c_by_word * nbits_in_data;
static const unsigned k_size_packed = k_h * k_w * k_c_by_word * nbits_k_data;

static const unsigned inb_h = k_h + 1;
static const unsigned inb_w = k_w + 1;
static const unsigned num_in_line = (max_in_w_with_pad * (k_h - 1)) + k_w;
static const bool has_thresholds = true;
} // namespace conv1x1_params

namespace conv_kn2row_params {

static const unsigned num_pe = conv_common_params::num_pe;
static const unsigned num_thresholds = conv_common_params::num_thresholds;

static const unsigned max_in_c = conv_common_params::max_in_c;
static const unsigned max_in_c_by_word = conv_common_params::max_in_c_by_word;
static const unsigned min_in_c = conv_common_params::min_in_c;
static const unsigned min_in_c_by_word = conv_common_params::min_in_c_by_word;
static const unsigned max_in_b = conv_common_params::max_in_b;
static const unsigned min_in_b = conv_common_params::min_in_b;

static const unsigned max_k_h = 3;
static const unsigned max_k_w = 3;
static const unsigned min_k_h = 1;
static const unsigned min_k_w = 1;

static const unsigned tile_h = 16;
static const unsigned tile_w = 32;

static const unsigned in_tile_h = tile_h + (max_k_h - 1) * 2;
static const unsigned in_tile_w = tile_w + (max_k_w - 1) * 2;
} // namespace conv_kn2row_params
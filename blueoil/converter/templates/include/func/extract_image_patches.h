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
=============================================================================*/

#ifndef DLK_FUNC_EXTRACT_IMAGE_PATCHES
#define DLK_FUNC_EXTRACT_IMAGE_PATCHES

#include <algorithm>
#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"
#include "pack_input_to_qwords.h"
#include <limits.h>

#ifdef USE_NEON
#include <arm_neon.h>
#endif

template <typename T>
void func_ExtractImagePatches(
    const TensorView<T, MemoryLayout::NHWC>& input,
    const TensorView<T, MemoryLayout::NHWC>& output,
    T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("ExtractImagePatches");

  const auto in_shape = input.get_shape();
  const T_UINT input_width = in_shape[2];
  const T_UINT input_channels = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[1];
  const T_UINT out_width = out_shape[2];
  const T_UINT out_channels = out_shape[3];

  for(T_UINT kz = 0; kz < input_channels; ++kz)
    for(T_UINT wi = 0; wi < out_height; wi++)
      for(T_UINT wj = 0; wj < out_width; wj++)
        for(T_UINT ki = 0; ki < kernel_size; ki++)
          for(T_UINT kj = 0; kj < kernel_size; kj++)
          {
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;
            const auto ch = kz + (ki * kernel_size + kj) * input_channels;
            const auto out_idx = wi * out_width * out_channels
              + wj * out_channels
              + ch;
            const auto in_idx = row * input_width * input_channels
              + col * input_channels
              + kz;
            output.data()[out_idx]
                = input.data()[in_idx];
          }

  Measurement::Stop();
}

inline void func_ExtractImagePatches(
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& output,
    T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("ExtractImagePatches");

  const auto in_shape = input.get_shape();
  const T_UINT input_width = in_shape[1];
  const T_UINT input_channels = in_shape[2];
  const T_UINT bits_per_input = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[0];
  const T_UINT out_width = out_shape[1];
  const T_UINT out_channels = out_shape[2];

  T_UINT output_index = 0;

  if (out_channels < kernel_size * kernel_size) {
    int bit_shift = out_channels * QUANTIZED_PACKED::BitCount / (kernel_size * kernel_size);
    const QUANTIZED_PACKED::base_t mask((QUANTIZED_PACKED::base_t(1) << bit_shift) - 1);
    std::fill(output.data(), output.data() + output.size(), QUANTIZED_PACKED(0));
    for(T_UINT wi = 0; wi < out_height; wi++)
      for(T_UINT wj = 0; wj < out_width; wj++)
        for(T_UINT ki = 0; ki < kernel_size; ki++)
          for(T_UINT kj = 0; kj < kernel_size; kj++)
          {
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;
            T_UINT ch = (ki * kernel_size + kj) * bit_shift;
            T_UINT ch_high = ch / QUANTIZED_PACKED::BitCount;
            T_UINT ch_low = ch % QUANTIZED_PACKED::BitCount;
#ifdef USE_NEON
            const auto out_idx = wi * out_width * out_channels * bits_per_input
              + wj * out_channels * bits_per_input
              + ch_high * bits_per_input;
            const auto in_idx = row * input_width * input_channels * bits_per_input
              + col * input_channels * bits_per_input;
            const auto in = vld1_u32(reinterpret_cast<uint32_t*>(input.data() + in_idx));
            const auto masked = vand_u32(vdup_n_u32(mask), in);
#ifdef AARCH32
            const auto shifted = vshl_u32(masked, vdup_n_s32(ch_low));
#else
            const auto shifted = vshl_n_u32(masked, ch_low);
#endif
            const auto out_old = vld1_u32(reinterpret_cast<uint32_t*>(output.data() + out_idx));
            const auto out_new = vorr_u32(out_old, shifted);
            vst1_u32(reinterpret_cast<uint32_t*>(output.data() + out_idx), out_new);
#else
            for(T_UINT digit = 0; digit < bits_per_input; ++digit) {
              const auto out_idx = wi * out_width * out_channels * bits_per_input
                + wj * out_channels * bits_per_input
                + ch_high * bits_per_input
                + digit;
              const auto in_idx = row * input_width * input_channels * bits_per_input
                + col * input_channels * bits_per_input
                + digit;
              output.data()[out_idx] |= QUANTIZED_PACKED((mask & input.data()[in_idx].Raw()) << ch_low);
            }
#endif
          }
  } else {
    for(T_UINT ih = 0; ih < input_channels; ++ih)
      for(T_UINT wi = 0; wi < out_height; wi++)
        for(T_UINT wj = 0; wj < out_width; wj++)
          for(T_UINT ki = 0; ki < kernel_size; ki++)
            for(T_UINT kj = 0; kj < kernel_size; kj++)
            {
              T_INT row = (wi * stride) + ki;
              T_INT col = (wj * stride) + kj;
#ifdef USE_NEON
              const auto ch_high = ih + (ki * kernel_size + kj) * input_channels;
              const auto out_idx = wi * out_width * out_channels * bits_per_input
                + wj * out_channels * bits_per_input
                + ch_high * bits_per_input;
              const auto in_idx = row * input_width * input_channels * bits_per_input
                + col * input_channels * bits_per_input
                + ih * bits_per_input;
              const auto in = vld1_u32(reinterpret_cast<uint32_t*>(input.data() + in_idx));
              vst1_u32(reinterpret_cast<uint32_t*>(output.data() + out_idx), in);
#else
              for(T_UINT digit = 0; digit < bits_per_input; ++digit) {
                const auto ch_high = ih + (ki * kernel_size + kj) * input_channels;
                const auto out_idx = wi * out_width * out_channels * bits_per_input
                  + wj * out_channels * bits_per_input
                  + ch_high * bits_per_input
                  + digit;
                const auto in_idx = row * input_width * input_channels * bits_per_input
                  + col * input_channels * bits_per_input
                  + ih * bits_per_input
                  + digit;
                output.data()[out_idx]
                  = input.data()[in_idx];
              }
#endif
            }
  }

  Measurement::Stop();
}

inline void func_ExtractImagePatches(
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    T_UINT kernel_size, T_UINT stride)
{
  Measurement::Start("ExtractImagePatches");
  const auto in_shape = input.get_shape();
  const T_UINT input_height = in_shape[1];
  const T_UINT input_width = in_shape[2];
  const T_UINT input_channels = in_shape[0];
  const T_UINT bits_per_input = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[1];
  const T_UINT out_width = out_shape[2];
  const T_UINT out_channels = out_shape[0];

  T_UINT output_index = 0;

  if (out_channels < kernel_size * kernel_size) {
    const T_UINT kernel_area = kernel_size * kernel_size;
    const T_UINT bit_shift = out_channels * QUANTIZED_PACKED::BitCount / kernel_area;
    const QUANTIZED_PACKED::base_t mask((QUANTIZED_PACKED::base_t(1) << bit_shift) - 1);
    const T_UINT lb_kernel_size = __builtin_ctz(kernel_size);
    const T_UINT kernel_mask = (1 << lb_kernel_size) - 1;
#ifdef USE_NEON
    const auto shift_ref = vcombine_s32(vdup_n_s32(0), vdup_n_s32(bit_shift));
    const auto add = vdupq_n_s32(bit_shift * 2);
    const auto mask_v = vdupq_n_u32(mask);
#else
    const uint64_t mask64 = mask * 0x1'0000'0001ull;
#endif
    const T_UINT blocks = kernel_area / out_channels;
#pragma omp parallel for
    for(T_UINT wi = 0; wi < out_height; wi++)
      for(T_UINT wj = 0; wj < out_width; wj++)
#ifdef USE_NEON
        for(T_UINT k = 0; k < out_channels; ++k) {
          auto tmp = vdupq_n_u32(0);
          auto shift = shift_ref;
          for(T_UINT i = 0; i < blocks; i += 2) {
            T_UINT ki = (k * blocks + i) >> lb_kernel_size;
            T_UINT kj = (k * blocks + i) & kernel_mask;
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;
            const auto in_idx = row * input_width * bits_per_input
              + col * bits_per_input;
            const auto in = vld1q_u32(reinterpret_cast<uint32_t*>(input.data() + in_idx));
            const auto masked = vandq_u32(mask_v, in);
            const auto shifted = vshlq_u32(masked, shift);
            shift += add;
            tmp |= shifted;
          }
          const auto out = vorr_u32(vget_low_u32(tmp), vget_high_u32(tmp));
          const auto out_idx = k * out_height * out_width * bits_per_input
            + wi * out_width * bits_per_input
            + wj * bits_per_input;
          vst1_u32(reinterpret_cast<uint32_t*>(output.data() + out_idx), out);
        }
#else
        for(T_UINT k = 0; k < out_channels; ++k) {
          uint64_t out = 0;
          for(T_UINT i = 0; i < blocks; ++i) {
            T_UINT ki = (k * blocks + i) >> lb_kernel_size;
            T_UINT kj = (k * blocks + i) & kernel_mask;
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;
            const auto in_idx = row * input_width * bits_per_input
              + col * bits_per_input;
            const auto in = *reinterpret_cast<uint64_t*>(input.data() + in_idx);
            out |= (mask64 & in) << (i * bit_shift);
          }
          const auto out_idx = k * out_height * out_width * bits_per_input
            + wi * out_width * bits_per_input
            + wj * bits_per_input;
          *reinterpret_cast<uint64_t*>(output.data() + out_idx) = out;
        }
#endif
  } else {
    for(T_UINT ih = 0; ih < input_channels; ++ih)
      for(T_UINT wi = 0; wi < out_height; wi++)
        for(T_UINT wj = 0; wj < out_width; wj++)
          for(T_UINT ki = 0; ki < kernel_size; ki++)
            for(T_UINT kj = 0; kj < kernel_size; kj++)
            {
              T_INT row = (wi * stride) + ki;
              T_INT col = (wj * stride) + kj;
              const auto ch_high = ih + (ki * kernel_size + kj) * input_channels;
              const auto out_idx = ch_high * out_height * out_width * bits_per_input
                + wi * out_width * bits_per_input
                + wj * bits_per_input;
              const auto in_idx = ih * input_height * input_width * bits_per_input
                + row * input_width * bits_per_input
                + col * bits_per_input;
#ifdef USE_NEON
              const auto in = vld1_u32(reinterpret_cast<uint32_t*>(input.data() + in_idx));
              vst1_u32(reinterpret_cast<uint32_t*>(output.data() + out_idx), in);
#else
              *reinterpret_cast<uint64_t*>(output.data() + out_idx) =
                  *reinterpret_cast<uint64_t*>(input.data() + in_idx);
#endif
            }
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_EXTRACT_IMAGE_PATCHES

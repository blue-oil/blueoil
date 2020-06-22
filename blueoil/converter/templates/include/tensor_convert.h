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
=============================================================================*/

#ifndef DLK_TENSOR_CONVERT_H_INCLUDED
#define DLK_TENSOR_CONVERT_H_INCLUDED

#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "func/impl/quantized_conv2d_tiling.h"
#ifdef USE_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

inline void convert_tensor(const TensorView<BIN_CONV_OUTPUT, MemoryLayout::HWC>& before,
    const TensorView<BIN_CONV_OUTPUT, MemoryLayout::ChHWCl>& after) {
  const auto in_shape = before.get_shape();
  const auto in_height = in_shape[0];
  const auto in_width = in_shape[1];
  const auto out_shape = after.get_shape();
  const auto channel_high = out_shape[0];
  const auto channel_low = out_shape[3];
  Measurement::Start("Convert Tensor");
  for (std::size_t dh = 0; dh < channel_high; ++dh)
    for (std::size_t r = 0; r < in_height; ++r)
      for (std::size_t c = 0; c < in_width; ++c)
        for (std::size_t dl = 0; dl < channel_low; ++dl)
          after(dh, r, c, dl) = before(r, c, dh * channel_low + dl);
  Measurement::Stop();
}

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& before,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& after) {
  const auto in_shape = before.get_shape();
  const auto height = in_shape[0];
  const auto width = in_shape[1];
  const auto channel = in_shape[2];
  const auto bits = in_shape[3];
  Measurement::Start("Convert Tensor");
#pragma omp parallel for
  for (std::size_t i = 0; i < height; ++i)
    for (std::size_t j = 0; j < width; ++j)
      for (std::size_t k = 0; k < channel; ++k) {
        const auto idx_before = i * width * channel * bits
            + j * channel * bits
            + k * bits;
        const auto idx_after = k * height * width * bits
            + i * width * bits
            + j * bits;
#ifdef AARCH32
        const auto tmp = vld1_u32(reinterpret_cast<uint32_t*>(before.data() + idx_before));
        vst1_u32(reinterpret_cast<uint32_t*>(after.data() + idx_after), tmp);
#else
        *reinterpret_cast<uint64_t*>(after.data() + idx_after) =
            *reinterpret_cast<uint64_t*>(before.data() + idx_before);
#endif
      }
  Measurement::Stop();
}

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& before,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& after) {
  const auto in_shape = before.get_shape();
  const auto height = in_shape[1];
  const auto width = in_shape[2];
  const auto channel = in_shape[0];
  const auto bits = in_shape[3];
  Measurement::Start("Convert Tensor");
#pragma omp parallel for
  for (std::size_t i = 0; i < height; ++i)
    for (std::size_t j = 0; j < width; ++j)
      for (std::size_t k = 0; k < channel; ++k) {
        const auto idx_before = k * height * width * bits
            + i * width * bits
            + j * bits;
        const auto idx_after = i * width * channel * bits
            + j * channel * bits
            + k * bits;
#ifdef AARCH32
        const auto tmp = vld1_u32(reinterpret_cast<uint32_t*>(before.data() + idx_before));
        vst1_u32(reinterpret_cast<uint32_t*>(after.data() + idx_after), tmp);
#else
        *reinterpret_cast<uint64_t*>(after.data() + idx_after) =
            *reinterpret_cast<uint64_t*>(before.data() + idx_before);
#endif
      }
  Measurement::Stop();
}

inline void convert_tensor(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& before,
    const dlk::impl::tiling_input_t& after) {
  dlk::impl::pack_input_for_tiling(before, after);
}

template <typename T, MemoryLayout layout>
void convert_tensor(const TensorView<T, layout>& before,
    const TensorView<T, layout>& after) {
  const auto num_elems = before.size();
  Measurement::Start("Convert Tensor");
#ifdef _OPENMP
  const auto num_threads = omp_get_max_threads();
  const auto chunk_size = (num_elems + num_threads - 1) / num_threads;
#pragma omp parallel for
  for (int i = 0; i < num_elems; i += chunk_size) {
    std::copy(before.data() + i, before.data() + std::min(i + chunk_size, num_elems), after.data() + i);
  }
#else
  std::copy(before.data(), before.data() + num_elems, after.data());
#endif
  Measurement::Stop();
}

#endif

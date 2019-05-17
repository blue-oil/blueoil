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

#ifndef DLK_TENSOR_CONVERT_H_INCLUDED
#define DLK_TENSOR_CONVERT_H_INCLUDED

#include "global.h"
#include "tensor_view.h"
#include "pack_input_to_qwords.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "func/impl/quantized_conv2d_tiling.h"
#include "func/impl/quantized_conv2d_dim2col.h"

inline void convert_tensor(const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& before,
    const dlk::impl::kn2row_input_t& after) {
  const auto in_shape = before.get_shape();
  const auto height = in_shape[0];
  const auto width = in_shape[1];
  const auto channel = in_shape[2];
  const auto bits = in_shape[3];
  for (std::size_t i = 0; i < height; ++i)
    for (std::size_t j = 0; j < width; ++j)
      for (std::size_t k = 0; k < channel; ++k)
        for (std::size_t d = 0; d < bits; ++d)
          after(k, i, j, d, 0) = before(i, j, k, d, 0);
}

inline void convert_tensor(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& before,
    const dlk::impl::tiling_input_t& after) {
  dlk::impl::pack_input_for_tiling(before, after);
}

inline void convert_tensor(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& before,
    const dlk::impl::dim2col_input_t& after,
    const binary_convolution_parameters& p) {
  dlk::impl::im2col(before, after, p);
}

inline void convert_tensor(const kernel_t& before,
    const dlk::impl::kn2row_kernel_t& after,
    const binary_convolution_parameters& p) {
  dlk::impl::quantized_ohwi_to_hwoi(before, after, p);
}

template <typename T, MemoryLayout layout>
void convert_tensor(const TensorView<T, layout>& before,
    const TensorView<T, layout>& after) {
  auto num_elems = before.size();
  std::copy(before.data(), before.data() + num_elems, after.data());
}

#endif

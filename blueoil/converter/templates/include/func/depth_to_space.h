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

#ifndef DLK_FUNC_DEPTH_TO_SPACE_H_INCLUDED
#define DLK_FUNC_DEPTH_TO_SPACE_H_INCLUDED

#include "types.h"
#include "time_measurement.h"
#include "tensor_view.h"

inline void func_DepthToSpace(const TensorView<float, MemoryLayout::NHWC>& input,
    const TensorView<float, MemoryLayout::NHWC>& output,
    T_UINT a, T_UINT b, T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("DepthToSpace");

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[1];
  const auto out_width = out_shape[2];
  const auto out_depth = out_shape[3];

  for(T_UINT wi = 0; wi < out_height; wi += stride)
    for(T_UINT wj = 0; wj < out_width; wj += stride)
    {
      for(T_UINT ki = 0; ki < kernel_size; ki++)
        for(T_UINT kj = 0; kj < kernel_size; kj++)
          for(T_UINT kz = 0; kz < out_depth; kz++)
          {
            T_INT row = wi + ki;
            T_INT col = wj + kj;

            T_INT idx = kz + ki * kernel_size * out_depth + kj * out_depth;
            output(0, row, col, kz) = input(0, wi/stride, wj/stride, idx);
          }
      }

  Measurement::Stop();
}

template <typename T>
void func_DepthToSpace(const TensorView<QuantizedPacked<T>, MemoryLayout::HWChBCl>& input,
    const TensorView<QuantizedPacked<T>, MemoryLayout::HWChBCl>& output,
    T_UINT a, T_UINT b, T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("DepthToSpace");

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[0];
  const auto out_width = out_shape[1];
  const auto bits = out_shape[3];
  const auto out_depth = out_shape[2];

  for(T_UINT wi = 0; wi < out_height; wi += stride)
    for(T_UINT wj = 0; wj < out_width; wj += stride)
    {
      for(T_UINT ki = 0; ki < kernel_size; ki++)
        for(T_UINT kj = 0; kj < kernel_size; kj++)
          for(T_UINT kz = 0; kz < out_depth; kz++)
          {
            T_INT row = wi + ki;
            T_INT col = wj + kj;

            T_INT idx = kz + ki * kernel_size * out_depth + kj * out_depth;
            for (T_INT digit = 0; digit < bits; ++digit) {
              output(row, col, kz, digit, 0) = input(wi/stride, wj/stride, idx, digit, 0);
            }
          }
      }

  Measurement::Stop();
}

template <typename T>
void func_DepthToSpace(const TensorView<QuantizedPacked<T>, MemoryLayout::ChHWBCl>& input,
    const TensorView<QuantizedPacked<T>, MemoryLayout::ChHWBCl>& output,
    T_UINT a, T_UINT b, T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("DepthToSpace");

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[1];
  const auto out_width = out_shape[2];
  const auto bits = out_shape[3];
  const auto out_depth = out_shape[0];

  for(T_UINT wi = 0; wi < out_height; wi += stride)
    for(T_UINT wj = 0; wj < out_width; wj += stride)
    {
      for(T_UINT ki = 0; ki < kernel_size; ki++)
        for(T_UINT kj = 0; kj < kernel_size; kj++)
          for(T_UINT kz = 0; kz < out_depth; kz++)
          {
            T_INT row = wi + ki;
            T_INT col = wj + kj;

            T_INT idx = kz + ki * kernel_size * out_depth + kj * out_depth;
            for (T_INT digit = 0; digit < bits; ++digit) {
              output(kz, row, col, digit, 0) = input(idx, wi/stride, wj/stride, digit, 0);
            }
          }
      }

  Measurement::Stop();
}

#endif // DLK_FUNC_DEPTH_TO_SPACE_H_INCLUDED

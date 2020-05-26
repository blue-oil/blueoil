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

#ifndef DLK_FUNC_RESIZE_NEAREST_NEIGHBOR_H_INCLUDED
#define DLK_FUNC_RESIZE_NEAREST_NEIGHBOR_H_INCLUDED

#include "types.h"
#include "time_measurement.h"
#include "tensor_view.h"


inline void func_ResizeNearestNeighbor(const TensorView<float, MemoryLayout::NHWC>& input,
    const TensorView<float, MemoryLayout::NHWC>& output) {

  Measurement::Start("ResizeNearestNeighbor");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[1];
  const auto out_width = out_shape[2];
  const auto out_depth = out_shape[3];

  const auto height_scale = (in_height << 16) / out_height + 1;
  const auto width_scale = (in_width << 16) / out_width + 1;

  for(T_UINT out_y = 0; out_y < out_height; out_y++) {
    T_INT in_y = std::min((out_y * height_scale) >> 16, in_height - 1);
    for(T_UINT out_x = 0; out_x < out_width; out_x++) {
      T_INT in_x = std::min((out_x * width_scale) >> 16, in_width - 1);
      for(T_UINT out_z = 0; out_z < out_depth; out_z++) {
        output(0, out_y, out_x, out_z) = input(0, in_y, in_x, out_z);
      }
    }
  }
  Measurement::Stop();
}


template <typename T>
void func_ResizeNearestNeighbor(const TensorView<QuantizedPacked<T>, MemoryLayout::HWChBCl>& input,
    const TensorView<QuantizedPacked<T>, MemoryLayout::HWChBCl>& output) {
  Measurement::Start("ResizeNearestNeighbor");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[0];
  const auto in_width = in_shape[1];

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[0];
  const auto out_width = out_shape[1];
  const auto bits = out_shape[3];
  const auto out_depth = out_shape[2];

  const auto height_scale = (in_height << 16) / out_height + 1;
  const auto width_scale = (in_width << 16) / out_width + 1;

  for(T_UINT out_y = 0; out_y < out_height; out_y++) {
    T_INT in_y = std::min((out_y * height_scale) >> 16, in_height - 1);
    for(T_UINT out_x = 0; out_x < out_width; out_x++) {
      T_INT in_x = std::min((out_x * width_scale) >> 16, in_width - 1);
      for(T_UINT out_z = 0; out_z < out_depth; out_z++) {
        for (T_INT digit = 0; digit < bits; ++digit) {
          output(out_y, out_x, out_z, digit, 0) = input(in_y, in_x, out_z, digit, 0);
        }
      }
    }
  }
  Measurement::Stop();
}


template <typename T>
void func_ResizeNearestNeighbor(const TensorView<QuantizedPacked<T>, MemoryLayout::ChHWBCl>& input,
    const TensorView<QuantizedPacked<T>, MemoryLayout::ChHWBCl>& output) {
  Measurement::Start("ResizeNearestNeighbor");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[1];
  const auto out_width = out_shape[2];
  const auto bits = out_shape[3];
  const auto out_depth = out_shape[0];

  const auto height_scale = (in_height << 16) / out_height + 1;
  const auto width_scale = (in_width << 16) / out_width + 1;

  for(T_UINT out_y = 0; out_y < out_height; out_y++) {
    T_INT in_y = std::min((out_y * height_scale) >> 16, in_height - 1);
    for(T_UINT out_x = 0; out_x < out_width; out_x++) {
      T_INT in_x = std::min((out_x * width_scale) >> 16, in_width - 1);
      for(T_UINT out_z = 0; out_z < out_depth; out_z++) {
        for (T_INT digit = 0; digit < bits; ++digit) {
          output(out_z, out_y, out_x, digit, 0) = input(out_z, in_y, in_x, digit, 0);
        }
      }
    }
  }
  Measurement::Stop();
}


#endif // DLK_FUNC_RESIZE_NEAREST_NEIGHBOR_H_INCLUDED

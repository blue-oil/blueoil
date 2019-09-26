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

#ifndef DLK_FUNC_CONCAT_ON_DEPTH_H_INCLUDED
#define DLK_FUNC_CONCAT_ON_DEPTH_H_INCLUDED

#include <tuple>
#include <type_traits>

#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"

namespace dlk {
namespace impl {

constexpr std::size_t index_channels_high(const MemoryLayout layout) {
  switch (layout) {
    case MemoryLayout::ChHWBCl: return 0;
    case MemoryLayout::HWChBCl: return 2;
    case MemoryLayout::NHWC: return 3;
    default: return 0;
  }
}

template<class T>
T access_ChHWBCl(const TensorView<T, MemoryLayout::ChHWBCl>& tensor,
    const std::size_t ch,
    const std::size_t h,
    const std::size_t w,
    const std::size_t digit) {
  return tensor(ch, h, w, digit, 0);
}

template<class T>
T access_ChHWBCl(const TensorView<T, MemoryLayout::HWChBCl>& tensor,
    const std::size_t ch,
    const std::size_t h,
    const std::size_t w,
    const std::size_t digit) {
  return tensor(h, w, ch, digit, 0);
}

template<class T>
T access_HWChBCl(const TensorView<T, MemoryLayout::ChHWBCl>& tensor,
    const std::size_t h,
    const std::size_t w,
    const std::size_t ch,
    const std::size_t digit) {
  return tensor(ch, h, w, digit, 0);
}

template<class T>
T access_HWChBCl(const TensorView<T, MemoryLayout::HWChBCl>& tensor,
    const std::size_t h,
    const std::size_t w,
    const std::size_t ch,
    const std::size_t digit) {
  return tensor(h, w, ch, digit, 0);
}

template<class T>
T access_NHWC(const TensorView<T, MemoryLayout::NHWC>& tensor,
    const std::size_t h,
    const std::size_t w,
    const std::size_t ch) {
  return tensor(0, h, w, ch);
}

template <typename TOut, std::size_t I, typename... TInputs>
class ConcatOnDepth;

template <typename TOut, std::size_t I, typename Enable, typename... TInputs>
struct ConcatOnDepthImpl;

template <typename TOut, std::size_t I, typename...TInputs>
struct ConcatOnDepthImpl<TOut, I, typename std::enable_if<(I < sizeof...(TInputs))>::type, TInputs...> {
  void operator()(const std::tuple<TInputs...>& inputs,
      const std::size_t stride_depth,
      const std::size_t offset_depth,
      const TensorView<TOut, MemoryLayout::ChHWBCl>& output) {
    const auto shape = output.get_shape();
    const auto out_height = shape[1];
    const auto out_width = shape[2];
    const auto bits = shape[3];
    const auto input = std::get<I>(inputs);
    const auto index = index_channels_high(decltype(input)::layout);
    const auto depth = input.get_shape()[index];
    for (std::size_t d = 0; d < depth; ++d) {
      for (std::size_t h = 0; h < out_height; ++h) {
        for (std::size_t w = 0; w < out_width; ++w) {
          for (std::size_t digit = 0; digit < bits; ++digit) {
            output(offset_depth + d, h, w, digit, 0)
              = access_ChHWBCl(input, d, h, w, digit);
          }
        }
      }
    }
    ConcatOnDepth<TOut, I+1, TInputs...> func;
    func(inputs, stride_depth, offset_depth + depth, output);
  }
  void operator()(const std::tuple<TInputs...>& inputs,
      const std::size_t stride_depth,
      const std::size_t offset_depth,
      const TensorView<TOut, MemoryLayout::HWChBCl>& output) {
    const auto shape = output.get_shape();
    const auto out_height = shape[0];
    const auto out_width = shape[1];
    const auto bits = shape[3];
    const auto input = std::get<I>(inputs);
    const auto index = index_channels_high(decltype(input)::layout);
    const auto depth = input.get_shape()[index];
    for (std::size_t d = 0; d < depth; ++d) {
      for (std::size_t h = 0; h < out_height; ++h) {
        for (std::size_t w = 0; w < out_width; ++w) {
          for (std::size_t digit = 0; digit < bits; ++digit) {
            output(h, w, offset_depth + d, digit, 0)
              = access_HWChBCl(input, h, w, d, digit);
          }
        }
      }
    }
    ConcatOnDepth<TOut, I+1, TInputs...> func;
    func(inputs, stride_depth, offset_depth + depth, output);
  }
  void operator()(const std::tuple<TInputs...>& inputs,
      const std::size_t stride_depth,
      const std::size_t offset_depth,
      const TensorView<TOut, MemoryLayout::NHWC>& output) {
    const auto shape = output.get_shape();
    const auto out_height = shape[1];
    const auto out_width = shape[2];
    const auto input = std::get<I>(inputs);
    const auto index = index_channels_high(decltype(input)::layout);
    const auto depth = input.get_shape()[index];
    for (std::size_t d = 0; d < depth; ++d) {
      for (std::size_t h = 0; h < out_height; ++h) {
        for (std::size_t w = 0; w < out_width; ++w) {
            output(0, h, w, offset_depth + d)
              = access_NHWC(input, h, w, d);
        }
      }
    }
    ConcatOnDepth<TOut, I+1, TInputs...> func;
    func(inputs, stride_depth, offset_depth + depth, output);
  }
};

template <typename TOut, std::size_t I, typename...TInputs>
struct ConcatOnDepthImpl<TOut, I, typename std::enable_if<!(I < sizeof...(TInputs))>::type, TInputs...> {
  void operator()(const std::tuple<TInputs...>& inputs,
      const std::size_t stride_depth,
      const std::size_t offset_depth,
      const TensorView<TOut, MemoryLayout::ChHWBCl>& output) {
    // nothing to do
  }
  void operator()(const std::tuple<TInputs...>& inputs,
      const std::size_t stride_depth,
      const std::size_t offset_depth,
      const TensorView<TOut, MemoryLayout::HWChBCl>& output) {
    // nothing to do
  }
  void operator()(const std::tuple<TInputs...>& inputs,
      const std::size_t stride_depth,
      const std::size_t offset_depth,
      const TensorView<TOut, MemoryLayout::NHWC>& output) {
    // nothing to do
  }
};

template <typename TOut, std::size_t I, typename... TInputs>
class ConcatOnDepth : public ConcatOnDepthImpl<TOut, I, void, TInputs...> {};

} // namespace impl
} // namespace detail

template<class... TInputs, class TOut, MemoryLayout output_layout>
void func_ConcatOnDepth(const std::tuple<TInputs...>& inputs,
    const TensorView<TOut, output_layout>& output) {
  Measurement::Start("func_ConcatOnDepth");
  const auto shape = output.get_shape();
  const auto index = dlk::impl::index_channels_high(output_layout);
  const auto depth = shape[index];
  dlk::impl::ConcatOnDepth<TOut, 0, TInputs...> func;
  func(inputs, depth, 0, output);
  Measurement::Stop();
}

#endif // DLK_FUNC_CONCAT_ON_DEPTH_H_INCLUDED

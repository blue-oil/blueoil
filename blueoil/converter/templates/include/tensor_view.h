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

#ifndef DLK_TENSOR_VIEW_H_INCLUDED
#define DLK_TENSOR_VIEW_H_INCLUDED

#include <cassert>
#include <array>
#include "types.h"

enum class MemoryLayout {
  Atom, // Scalar object
  C, // Channel
  N, // Batch size (or Output Channel)
  NC, // Batch, Channel
  WC, // Width, Channel
  TC, // Table, Channel
  HWC, // Height, Width, Channel
  NHWC, // Batch, Height, Width, Channel
  OHWI, // Output Channel, Height, Width, Input Channel
  WNC, // Width, Batch, Channel
  HWNC, // Height, Width, Batch, Channel
  HWOI, // Height, Width, Output Channel, Input Channel
  ChHWCl, // Channel (Higher dimension), Height, Width, Channel (Lower dimension)
  HWChBCl, // Height, Width, Channel (Higher dimension), Bit digit, Channel (Lower dimension)
  ChHWBCl, // Channel (Higher dimension), Height, Width, Bit digit, Channel (Lower dimension)
  OhIhHWOlIl, // Output Channel (Higher dimension), Input Channel (Higher dimension), Height, Width, Output Channel (Lower dimension), Input Channel (Lower dimension)
  Im2col, // im2col layout
  Padding, // for Padding. This is workaround, should be removed.
  Invalid, // Invalid Layout
};

constexpr std::size_t get_dim(const MemoryLayout& layout) {
  switch (layout) {
    case MemoryLayout::Atom: return 0;
    case MemoryLayout::C: return 1;
    case MemoryLayout::N: return 1;
    case MemoryLayout::NC: return 2;
    case MemoryLayout::WC: return 2;
    case MemoryLayout::TC: return 2;
    case MemoryLayout::HWC: return 3;
    case MemoryLayout::NHWC: return 4;
    case MemoryLayout::OHWI: return 4;
    case MemoryLayout::WNC: return 3;
    case MemoryLayout::HWNC: return 4;
    case MemoryLayout::HWOI: return 4;
    case MemoryLayout::ChHWCl: return 4;
    case MemoryLayout::HWChBCl: return 5;
    case MemoryLayout::ChHWBCl: return 5;
    case MemoryLayout::OhIhHWOlIl: return 6;
    case MemoryLayout::Im2col: return 4;
    case MemoryLayout::Padding: return 2;
    default: return 0;
  }
}

template <MemoryLayout layout>
constexpr std::size_t dim = get_dim(layout);

constexpr bool is_lower_dim(const MemoryLayout& lhs, const MemoryLayout& rhs) {
  return get_dim(lhs) < get_dim(rhs);
}

constexpr MemoryLayout inner_layout(const MemoryLayout& layout) {
  switch (layout) {
    case MemoryLayout::C: return MemoryLayout::Atom;
    case MemoryLayout::N: return MemoryLayout::Atom;
    case MemoryLayout::NC: return MemoryLayout::C;
    case MemoryLayout::WC: return MemoryLayout::C;
    case MemoryLayout::HWC: return MemoryLayout::WC;
    case MemoryLayout::NHWC: return MemoryLayout::HWC;
    case MemoryLayout::WNC: return MemoryLayout::NC;
    case MemoryLayout::HWNC: return MemoryLayout::WNC;
    default: return MemoryLayout::Invalid;
  }
}

template <typename T, MemoryLayout memory_layout>
class TensorView {
 public:
  using base_t = T;
  static constexpr auto layout = memory_layout;
  template <typename U>
  using tensor_info_t = std::array<U, dim<layout>>;
  TensorView(base_t* const ptr,
    const tensor_info_t<std::size_t>& shape)
    : ptr(ptr), shape(shape) {}
  template <typename... Ts>
  base_t& operator()(Ts&&... args) const {
    return *data(args...);
  }
  template <typename... Ts>
  base_t* data(Ts&&... args) const {
    static_assert(sizeof...(Ts) == dim<layout>, "Unmatched dimension");
    return ptr + get_offset(args...);
  }
  base_t* data() const {
    return ptr;
  }
  template <typename... Ts>
  std::size_t get_offset(Ts&&... args) const {
    tensor_info_t<std::size_t> offsets = {static_cast<std::size_t>(args)...};
    for (std::size_t i = sizeof...(Ts); i < dim<layout>; ++i) {
      offsets[i] = 0;
    }
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::ptrdiff_t i = dim<layout> - 1; i >= 0; --i) {
      assert(offsets[i] < shape[i]);
      offset += offsets[i] * stride;
      stride *= shape[i];
    }
    return offset;
  }
  template <std::size_t N>
  std::size_t get_offset_ary(const std::array<std::size_t, N>& arg) const {
    tensor_info_t<std::size_t> offsets;
    for (std::size_t i = 0; i < std::min(dim<layout>, N); ++i) {
      offsets[i] = arg[i];
    }
    for (std::size_t i = N; i < dim<layout>; ++i) {
      offsets[i] = 0;
    }
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::ptrdiff_t i = dim<layout> - 1; i >= 0; --i) {
      assert(offsets[i] < shape[i]);
      offset += offsets[i] * stride;
      stride *= shape[i];
    }
    return offset;
  }
  const tensor_info_t<std::size_t>& get_shape() const {
    return shape;
  }
  std::size_t size() const {
    std::size_t prod = 1;
    for (std::size_t i = 0; i < dim<layout>; ++i) {
      prod *= shape[i];
    }
    return prod;
  }
 private:
  base_t* ptr;
  tensor_info_t<std::size_t> shape;
};

template <typename T, MemoryLayout memory_layout>
class TensorView<QuantizedPacked<T>, memory_layout> {
 public:
  using base_t = QuantizedPacked<T>;
  static constexpr auto layout = memory_layout;
  template <typename U>
  using tensor_info_t = std::array<U, dim<layout>>;
  TensorView(base_t* const ptr,
    const tensor_info_t<std::size_t>& shape)
    : ptr(ptr), shape(shape) {}
  template <typename... Ts>
  base_t& operator()(Ts&&... args) const {
    return *data(args...);
  }
  template <typename... Ts>
  base_t* data(Ts&&... args) const {
    static_assert(sizeof...(Ts) == dim<layout>, "Unmatched dimension");
    return ptr + get_offset(args...);
  }
  base_t* data() const {
    return ptr;
  }
  template <typename... Ts>
  std::size_t get_offset(Ts&&... args) const {
    tensor_info_t<std::size_t> offsets = {static_cast<std::size_t>(args)...};
    for (std::size_t i = sizeof...(Ts); i < dim<layout>; ++i) {
      offsets[i] = 0;
    }
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::ptrdiff_t i = dim<layout> - 1; i >= 0; --i) {
      if (i == dim<layout> - 1) {
        const std::size_t real_shape = (shape[i] + base_t::BitCount - 1) / base_t::BitCount;
        assert(offsets[i] < real_shape);
        offset += offsets[i];
        stride *= real_shape;
      } else {
        assert(offsets[i] < shape[i]);
        offset += offsets[i] * stride;
        stride *= shape[i];
      }
    }
    return offset;
  }
  template <std::size_t N>
  std::size_t get_offset_ary(const std::array<std::size_t, N>& arg) const {
    tensor_info_t<std::size_t> offsets;
    for (std::size_t i = 0; i < std::min(dim<layout>, N); ++i) {
      offsets[i] = arg[i];
    }
    for (std::size_t i = N; i < dim<layout>; ++i) {
      offsets[i] = 0;
    }
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::ptrdiff_t i = dim<layout> - 1; i >= 0; --i) {
      if (i == dim<layout> - 1) {
        const std::size_t real_shape = (shape[i] + base_t::BitCount - 1) / base_t::BitCount;
        assert(offsets[i] < real_shape);
        offset += offsets[i];
        stride *= real_shape;
      } else {
        assert(offsets[i] < shape[i]);
        offset += offsets[i] * stride;
        stride *= shape[i];
      }
    }
    return offset;
  }
  const tensor_info_t<std::size_t>& get_shape() const {
    return shape;
  }
  std::size_t size() const {
    std::size_t prod = 1;
    for (std::size_t i = 0; i < dim<layout>; ++i) {
      if (i == dim<layout> - 1) {
        prod *= (shape[i] + base_t::BitCount - 1) / base_t::BitCount;
      } else {
        prod *= shape[i];
      }
    }
    return prod;
  }
 private:
  base_t* ptr;
  tensor_info_t<std::size_t> shape;
};

#endif

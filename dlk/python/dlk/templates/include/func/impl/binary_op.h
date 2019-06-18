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

#ifndef DLK_FUNC_IMPL_BINARY_OP_H_INCLUDED
#define DLK_FUNC_IMPL_BINARY_OP_H_INCLUDED

#include <cassert>
#include <cstddef>

#include <array>
#include <utility>
#include <type_traits>

#include "tensor_view.h"

namespace dlk {

namespace impl {

constexpr MemoryLayout get_layout(const MemoryLayout& me, const MemoryLayout& other) {
  if (is_lower_dim(me, other)) {
    return me;
  } else {
    return inner_layout(me);
  }
}

constexpr MemoryLayout output_layout(const MemoryLayout& lhs, const MemoryLayout& rhs) {
  if (is_lower_dim(lhs, rhs)) {
    return rhs;
  } else if (is_lower_dim(rhs, lhs)) {
    return lhs;
  } else if (lhs == rhs) {
    return lhs;
  } else {
    return MemoryLayout::Invalid;
  }
}

template <MemoryLayout layout_me, MemoryLayout layout_other>
auto next_info(const std::array<std::size_t, get_dim(layout_me)>& outer_info) {
  std::array<std::size_t, get_dim(get_layout(layout_me, layout_other))> res;
  for (std::size_t i = 0; i < res.size(); ++i) {
    res[i] = outer_info[i+(outer_info.size() - res.size())];
  }
  return res;
}

template <std::size_t N>
auto inner_info(const std::array<std::size_t, N>& outer_info) {
  std::array<std::size_t, N-1> res;
  for (std::size_t i = 0; i < N-1; ++i) {
    res[i] = outer_info[i+1];
  }
  return res;
}

template <MemoryLayout layout_me, MemoryLayout layout_other>
constexpr std::size_t reduce_dim = get_dim(layout_me) - get_dim(get_layout(layout_me, layout_other));

template <typename T, MemoryLayout layout_l, MemoryLayout layout_r, typename F>
struct binary_op {
  void operator()(const TensorView<T, layout_l>& lhs,
      const TensorView<T, layout_r>& rhs,
      const TensorView<T, output_layout(layout_l, layout_r)>& output,
      F f) const {
    static_assert(get_dim(layout_l) != get_dim(layout_r), "Unmatched Memory Layout");
    constexpr MemoryLayout layout = output_layout(layout_l, layout_r);
    const auto shape_l = lhs.get_shape();
    const auto shape_r = rhs.get_shape();
    const auto shape_out = output.get_shape();
    const auto next_shape_l = next_info<layout_l, layout_r>(shape_l);
    const auto next_shape_r = next_info<layout_r, layout_l>(shape_r);
    const auto next_shape_out = inner_info(shape_out);
    const auto loop = [&] {
      if (shape_l.size() < shape_r.size()) {
        return shape_r[0];
      } else {
        return shape_l[0];
      }
    }();
    binary_op<T, get_layout(layout_l, layout_r), get_layout(layout_r, layout_l), F> inner_op;
    for (std::size_t i = 0; i < loop; ++i) {
      std::array<std::size_t, reduce_dim<layout_l, layout_r>> index_array_l;
      if (index_array_l.size()) index_array_l[0] = i;
      const auto offset_l = lhs.get_offset_ary(index_array_l);
      using inner_l_t = TensorView<T, get_layout(layout_l, layout_r)>;
      inner_l_t inner_l(lhs.data() + offset_l, next_shape_l);
      std::array<std::size_t, reduce_dim<layout_r, layout_l>> index_array_r;
      if (index_array_r.size()) index_array_r[0] = i;
      const auto offset_r = rhs.get_offset_ary(index_array_r);
      using inner_r_t = TensorView<T, get_layout(layout_r, layout_l)>;
      inner_r_t inner_r(rhs.data() + offset_r, next_shape_r);
      const auto offset_out = output.get_offset(i);
      using inner_out_t = TensorView<T, inner_layout(layout)>;
      inner_out_t inner_out(output.data() + offset_out, next_shape_out);
      inner_op(inner_l, inner_r, inner_out, f);
    }
  }
};

template <typename T, MemoryLayout layout, typename F>
struct binary_op<T, layout, layout, F> {
  void operator()(const TensorView<T, layout>& lhs,
      const TensorView<T, layout>& rhs,
      const TensorView<T, layout>& output,
      F f) const {
    const auto shape_l = lhs.get_shape();
    const auto shape_r = rhs.get_shape();
    const auto shape_out = output.get_shape();
    constexpr MemoryLayout in_layout = inner_layout(layout);
    using inner_t = TensorView<T, in_layout>;
    const binary_op<T, in_layout, in_layout, F> inner_op;
    if (shape_l[0] == 1) {
      for (std::size_t i = 0; i < shape_r[0]; ++i) {
        inner_t inner_l(lhs.data(), inner_info(shape_l));
        const auto offset_r = rhs.get_offset(i);
        inner_t inner_r(rhs.data() + offset_r, inner_info(shape_r));
        const auto offset_out = output.get_offset(i);
        inner_t inner_out(output.data() + offset_out, inner_info(shape_out));
        inner_op(inner_l, inner_r, inner_out, f);
      }
    } else if (shape_r[0] == 1) {
      for (std::size_t i = 0; i < shape_l[0]; ++i) {
        const auto offset_l = lhs.get_offset(i);
        inner_t inner_l(lhs.data() + offset_l, inner_info(shape_l));
        inner_t inner_r(rhs.data(), inner_info(shape_r));
        const auto offset_out = output.get_offset(i);
        inner_t inner_out(output.data() + offset_out, inner_info(shape_out));
        inner_op(inner_l, inner_r, inner_out, f);
      }
    } else {
      assert(shape_l[0] == shape_r[0]);
      for (std::size_t i = 0; i < shape_r[0]; ++i) {
        const auto offset_l = lhs.get_offset(i);
        inner_t inner_l(lhs.data() + offset_l, inner_info(shape_l));
        const auto offset_r = rhs.get_offset(i);
        inner_t inner_r(rhs.data() + offset_r, inner_info(shape_r));
        const auto offset_out = output.get_offset(i);
        inner_t inner_out(output.data() + offset_out, inner_info(shape_out));
        inner_op(inner_l, inner_r, inner_out, f);
      }
    }
  }
};

template <typename T, typename F>
struct binary_op<T, MemoryLayout::Atom, MemoryLayout::Atom, F> {
  void operator()(const TensorView<T, MemoryLayout::Atom>& lhs,
      const TensorView<T, MemoryLayout::Atom>& rhs,
      const TensorView<T, MemoryLayout::Atom>& output,
      F f) const {
    output() = f(lhs(), rhs());
  }
};

} // namespace impl

} // namespace dlk

#endif // DLK_FUNC_IMPL_BINARY_OP_H_INCLUDED

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

#ifndef DLK_FUNC_IMPL_UNARY_OP_H_INCLUDED
#define DLK_FUNC_IMPL_UNARY_OP_H_INCLUDED

#include <cassert>
#include <cstddef>

#include "tensor_view.h"

namespace dlk {

namespace impl {

template <typename T, MemoryLayout layout, typename F>
void unary_op(const TensorView<T, layout>& input,
    const TensorView<T, layout>& output,
    F f) {
  assert(input.get_shape() == output.get_shape());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output.data()[i] = f(input.data()[i]);
  }
}

} // namespace impl

} // namespace dlk

#endif // DLK_FUNC_IMPL_UNARY_OP_H_INCLUDED

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

#ifndef DLK_FUNC_RELU_H_INCLUDED
#define DLK_FUNC_RELU_H_INCLUDED

#include "tensor_view.h"
#include "func/impl/unary_op.h"

template <typename T>
T relu(const T& x) { return std::max(x, T(0)); }

template <typename T, MemoryLayout layout>
void func_Relu(const TensorView<T, layout>& input,
    const TensorView<T, layout>& output) {
  Measurement::Start("ReLu");

  dlk::impl::unary_op(input, output, relu<T>);

  Measurement::Stop();
}

#endif // DLK_FUNC_RELU_H_INCLUDED

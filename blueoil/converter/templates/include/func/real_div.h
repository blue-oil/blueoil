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

#ifndef DLK_FUNC_REAL_DIV_H
#define DLK_FUNC_REAL_DIV_H

#include "tensor_view.h"
#include "func/impl/binary_op.h"
#include "time_measurement.h"

template <typename T, MemoryLayout layout_l, MemoryLayout layout_r>
void func_RealDiv(const TensorView<T, layout_l>& lhs,
    const TensorView<T, layout_r>& rhs,
    const TensorView<T, dlk::impl::output_layout(layout_l, layout_r)>& output) {
  Measurement::Start("RealDiv");

  dlk::impl::binary_op<T, layout_l, layout_r, std::divides<T>> bin_op;
  bin_op(lhs, rhs, output, std::divides<T>());

  Measurement::Stop();
}

#endif // DLK_FUNC_REAL_DIV_H

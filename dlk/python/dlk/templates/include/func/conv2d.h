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

#ifndef DLK_FUNC_CONV2D_H_INCLUDED
#define DLK_FUNC_CONV2D_H_INCLUDED

#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "tensor_view.h"

void func_Conv2D(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& weights,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    struct convolution_parameters p);


#endif // DLK_FUNC_CONV2D_H_INCLUDED

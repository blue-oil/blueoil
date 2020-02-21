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

#ifndef DLK_TENSOR_SAVE_H_INCLUDED
#define DLK_TENSOR_SAVE_H_INCLUDED

#include <string>

#include "types.h"
#include "global.h"
#include "tensor_view.h"

void save_tensor(TensorView<T_FLOAT, MemoryLayout::NHWC>&, const std::string& name, int32_t suffix);
void save_tensor(TensorView<T_FLOAT, MemoryLayout::C>&, const std::string& name, int32_t suffix);
void save_tensor(TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>&, const std::string& name, int32_t suffix);
void save_tensor(TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>&, const std::string& name, int32_t suffix);

#endif // DLK_TENSOR_SAVE_H_INCLUDED

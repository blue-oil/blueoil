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

#ifndef INPUT_{{ node.name }}_H_INCLUDED
#define INPUT_{{ node.name }}_H_INCLUDED

#include "global.h"

{% if node.transposed_data -%}

#ifdef RUN_ON_FPGA
extern const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.transposed_dimension_format }}> {{ node.name }}_output;
#elif defined USE_NEON || defined USE_AVX
extern const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension}}> {{ node.name }}_output;
#else
extern const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.kn2row_dimension_format }}> {{ node.name }}_output;
#endif

{% else -%}

extern const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension}}> {{ node.name }}_output;

{%- endif %}


#endif //{{ name }}_H_INCLUDED


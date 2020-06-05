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

#include "global.h"
#include "tensor_view.h"
#include "inputs/{{ node.name }}.h"

{% if node.transposed_data %}

#ifdef RUN_ON_FPGA
alignas(16) static unsigned char {{ node.name }}_raw[] = {
  {% for d in node.transposed_data -%}
  {{- pack_to_bytes(d) -}},
  {%- endfor %}
};
static constexpr decltype({{ node.name }}_output)::tensor_info_t<std::size_t> {{ node.name }}_shape = {
  {% for l in node.transposed_shape -%}
  {{- l -}},
  {%- endfor %}
};
const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.transposed_dimension_format }}> {{ node.name }}_output(
    reinterpret_cast<{{ node.dtype.cpptype() }}*>({{ node.name }}_raw),
    {{ node.name }}_shape);

#elif defined USE_NEON || defined USE_AVX
alignas(16) static unsigned char {{ node.name }}_raw[] = {
  {% for d in node.data.flatten() -%}
  {{- pack_to_bytes(d) -}},
  {%- endfor %}
};
static constexpr decltype({{ node.name }}_output)::tensor_info_t<std::size_t> {{ node.name }}_shape = {
  {% for l in node.shape -%}
  {{- l -}},
  {%- endfor %}
};
const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}> {{ node.name }}_output(
    reinterpret_cast<{{ node.dtype.cpptype() }}*>({{ node.name }}_raw),
    {{ node.name }}_shape);

#else
alignas(16) static unsigned char {{ node.name }}_raw[] = {
  {% for d in node.kn2row_data -%}
  {{- pack_to_bytes(d) -}},
  {%- endfor %}
};
static constexpr decltype({{ node.name }}_output)::tensor_info_t<std::size_t> {{ node.name }}_shape = {
  {% for l in node.kn2row_shape -%}
  {{- l -}},
  {%- endfor %}
};
const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.kn2row_dimension_format }}> {{ node.name }}_output(
    reinterpret_cast<{{ node.dtype.cpptype() }}*>({{ node.name }}_raw),
    {{ node.name }}_shape);
#endif

{% else -%}

alignas(16) static unsigned char {{ node.name }}_raw[] = {
  {% for d in node.data.flatten() -%}
  {{- pack_to_bytes(d) -}},
  {%- endfor %}
};
static constexpr decltype({{ node.name }}_output)::tensor_info_t<std::size_t> {{ node.name }}_shape = {
  {% for l in node.shape -%}
  {{- l -}},
  {%- endfor %}
};
const TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}> {{ node.name }}_output(
    reinterpret_cast<{{ node.dtype.cpptype() }}*>({{ node.name }}_raw),
    {{ node.name }}_shape);

{% endif %}

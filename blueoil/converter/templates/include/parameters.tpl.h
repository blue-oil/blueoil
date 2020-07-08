/* Copyright 2020 The Blueoil Authors. All Rights Reserved.

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

#ifndef PARAMETERS_H_INCLUDED
#define PARAMETERS_H_INCLUDED

#include <cstddef>


{%- if config.activate_hard_quantization %}
#define HARD_QUANTIZATION_ACTIVE
{% endif %}

{%- if config.threshold_skipping %}
#define THRESHOLD_SKIPPING_ACTIVE
{% endif %}

constexpr std::size_t MAX_SIZE_INPUTS_PER_LAYER = {{ params.max_size_inputs_per_layer }};
constexpr std::size_t MAX_SIZE_QINPUTS_PER_LAYER = {{ params.max_size_qinputs_per_layer }};
constexpr std::size_t MAX_SIZE_KN2ROW_BUFFER_PER_LAYER = {{ params.max_size_kn2row_buffer_per_layer }};
constexpr std::size_t MAX_SIZE_KN2ROW_COL_BLOCK = {{ params.max_size_kn2row_col_block }};

constexpr std::size_t MAX_SIZE_KERNELS_PER_LAYER = {{ params.max_size_kernels_per_layer }};
constexpr std::size_t MAX_SIZE_QKERNELS_PER_LAYER = {{ params.max_size_qkernels_per_layer }};
constexpr std::size_t MAX_SIZE_QKERNELS_PER_PE = {{ params.max_size_qkernels_per_pe }};

constexpr std::size_t MAX_SIZE_OUTPUTS_PER_LAYER = {{ params.max_size_outputs_per_layer }};
constexpr std::size_t MAX_SIZE_QOUTPUTS_PER_LAYER = {{ params.max_size_qoutputs_per_layer }};

#endif // PARAMETERS_H_INCLUDED

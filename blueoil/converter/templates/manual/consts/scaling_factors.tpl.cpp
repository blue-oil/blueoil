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
=============================================================================*/

#include "global.h"
#include "scaling_factors.h"

namespace scaling_factors {

{% for conv in quantized_convs -%}

{% if conv.quantizer.op_type == 'BinaryMeanScalingQuantizer' -%}

T_FLOAT {{ conv.name }} = {{ conv.quantizer.scaling_factor }};

{% elif conv.quantizer.op_type == 'BinaryChannelWiseMeanScalingQuantizer' -%}

T_FLOAT {{ conv.name }}[{{ conv.channels }}] = {
  {% for f in conv.quantizer.scaling_factor -%}
  {{- f -}},
  {%- endfor %}
};

{% else -%}

OtherQuantizerScalingFactorNotImplemented

{%- endif %}

{%- endfor %}

} // namespace scaling_factors

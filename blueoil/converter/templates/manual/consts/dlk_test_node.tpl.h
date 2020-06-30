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

#ifndef DLK_TEST_{{ test_node.name }}_H_INCLUDED
#define DLK_TEST_{{ test_node.name }}_H_INCLUDED

#include "global.h"

namespace dlk_test
{

{% if test_node.is_scalar -%}

extern {{ test_node.dtype }} dlk_test::{{ test_node.name }};

{% else -%}

extern {{ test_node.dtype }} {{ test_node.name }}[{{ test_node.shape_str }}];

{%- endif %}

}


#endif //{{ name }}_H_INCLUDED


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

#ifndef DLK_PACK_INPUT_TO_QWORDS_H_INCLUDED
#define DLK_PACK_INPUT_TO_QWORDS_H_INCLUDED

#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later

void pack_input_to_qwords(QUANTIZED_NOT_PACKED input[],
                          QUANTIZED_PACKED output[],
                          unsigned int len,
                          unsigned int input_bitwidth);

void pack_input_to_qwords(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED output[],
  struct binary_convolution_parameters bcp);


int pack_input(QUANTIZED_NOT_PACKED input[], size_t input_height, size_t input_width, size_t input_depth,
  size_t bits_per_input, QUANTIZED_PACKED output[]);

#endif // DLK_PACK_INPUT_TO_QWORDS_H_INCLUDED

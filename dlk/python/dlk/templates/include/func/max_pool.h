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

#ifndef DLK_FUNC_MAX_POOLING_H_INCLUDED
#define DLK_FUNC_MAX_POOLING_H_INCLUDED

#include "global.h"

struct max_pooling_parameters {
  T_UINT input_height;
  T_UINT input_width;
  T_UINT input_depth;
  T_UINT output_channels;
  T_UINT output_height;
  T_UINT output_width;
  T_UINT output_elements;
  T_UINT kernel_depth;
  T_UINT kernel_height;
  T_UINT kernel_width;
  T_UINT stride;
  T_UINT padding;
};

struct MaxPoolWithArgmax_parameters {
  T_UINT input_height;
  T_UINT input_width;
  T_UINT output_channels;
  T_UINT output_height;
  T_UINT output_width;
  T_UINT output_elements;
  T_UINT kernel_depth;
  T_UINT kernel_height;
  T_UINT kernel_width;
  T_UINT stride;
  T_UINT padding;
};

void func_MaxPool(T_FLOAT input[], T_FLOAT output[], struct max_pooling_parameters mpp, T_UINT out_height, T_UINT out_width, T_UINT out_depth);

void func_MaxPool(T_INT input[], T_INT output[], struct max_pooling_parameters mpp, T_UINT out_height, T_UINT out_width, T_UINT out_depth);

void func_MaxPoolWithArgmax(Quantized_t input[], Quantized_t output[], T_UINT indices[], struct MaxPoolWithArgmax_parameters mpp, T_UINT out_height, T_UINT out_width, T_UINT out_depth);

#endif // DLK_FUNC_MAX_POOLING_H_INCLUDED

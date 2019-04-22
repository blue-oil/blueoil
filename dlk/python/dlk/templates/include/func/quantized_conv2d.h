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

#ifndef DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED
#define DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED

#include <vector>
#include "operators.h"
#include "time_measurement.h"


void func_QuantizedConv2D(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED_KERNEL kernel[],
  T_FLOAT output[],
  T_FLOAT scaling_factor,
  binary_convolution_parameters p
);

void func_QuantizedConv2D(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED_KERNEL kernel[],
  T_FLOAT output[],
  T_FLOAT scaling_factor[],
  binary_convolution_parameters p
);

void func_QuantizedConv2DWithThreshold(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED_KERNEL kernel[],
  QUANTIZED_NOT_PACKED output[],
  T_FLOAT scaling_factor,
  binary_convolution_parameters p
);

void func_QuantizedConv2DWithThreshold(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED_KERNEL kernel[],
  QUANTIZED_NOT_PACKED output[],
  T_FLOAT scaling_factor[],
  binary_convolution_parameters p
);

void func_QuantizedConv2DWithThreshold(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED_KERNEL kernel[],
  T_FLOAT output[],
  T_FLOAT scaling_factor,
  binary_convolution_parameters p
);

void func_QuantizedConv2DWithThreshold(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED_KERNEL kernel[],
  T_FLOAT output[],
  T_FLOAT scaling_factor[],
  binary_convolution_parameters p
);


#endif // DLK_FUNC_QUANTIZED_CONV2D_H_INCLUDED



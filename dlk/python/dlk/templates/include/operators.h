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

#ifndef OPERATORS_HEADER
#define OPERATORS_HEADER

#include "dma_buffer.h"
#include "global.h"
#include <string>

struct convolution_parameters {
  T_UINT input_height;
  T_UINT input_width;
  T_UINT output_channels;
  T_UINT output_height;
  T_UINT output_width;
  T_UINT kernel_elements;
  T_UINT kernel_depth;
  T_UINT kernel_height;
  T_UINT kernel_width;
  T_UINT stride_along_height;
  T_UINT stride_along_width;
  T_UINT padding;
};

struct binary_convolution_parameters {
  struct convolution_parameters normal_conv_params;
  T_UINT bin_input_ndata;
  T_UINT bin_input_nwords;
  T_UINT bin_input_extra_bits;
  T_UINT bin_input_bitwidth;
  T_UINT bin_kernel_ndata;
  T_UINT layer_index;
  QUANTIZED_PACKED *device_input_buf;
  BIN_CONV_OUTPUT *device_output_buf;
  void print_device_output_buf(const std::string message) {
    std::cout << message << std::endl;
    for (int i = 0; i < 4; i++) {
      std::cout << device_output_buf[i] << std::endl;
    }
  }
  BIN_CONV_OUTPUT *thresholds;
  T_INT n_bit;
  T_FLOAT max_value;
  unsigned long device_input_phys_addr;
  unsigned long device_output_phys_addr;
  unsigned long device_kernel_phys_addr;

  DMA_Buffer *dma_input_buffer;
  DMA_Buffer *dma_output_buffer;
  const char* debug_name;
};

#endif

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

#ifndef GLOBAL_H
#define GLOBAL_H

#include <climits>
#include <inttypes.h>
#include <limits>
#include <stdlib.h>
#include "types.h"

#if defined RUN_ON_FPGA
#define VOLATILE_IF_FPGA volatile
  using QUANTIZED_PACKED = QuantizedPacked<volatile {{ params.default_qword_dtype.cpptype() }}>;
#else
#define VOLATILE_IF_FPGA
  using QUANTIZED_PACKED = QuantizedPacked<{{ params.default_qword_dtype.cpptype() }}>;
#endif
using QUANTIZED_PACKED_KERNEL = QuantizedPacked<{{ params.default_qword_dtype.cpptype() }}>;

#ifdef AARCH32
#define IP_CSR_ADDR 0xFF200000
#define HW_BASE_ADDR 0x20000000
#else
#define IP_CSR_ADDR 0xB0000000
#define HW_BASE_ADDR 0x20000000
#endif

#define INPUT_OFFSET 0x00000000
#define OUTPUT_OFFSET 0x08000000
#define KERNEL_OFFSET 0x18000000
#define THRESHOLD_OFFSET 0x1F000000

#define INPUT_ADDR (HW_BASE_ADDR+INPUT_OFFSET)
#define OUTPUT_ADDR (HW_BASE_ADDR+OUTPUT_OFFSET)
#define KERNEL_ADDR (HW_BASE_ADDR+KERNEL_OFFSET)
#define THRESHOLD_ADDR (HW_BASE_ADDR+THRESHOLD_OFFSET)

{%- if config.activate_hard_quantization %}
#define HARD_QUANTIZATION_ACTIVE
{% endif %}

{%- if config.threshold_skipping %}
#define THRESHOLD_SKIPPING_ACTIVE
{% endif %}

#define NUM_OF_A2W1_THRESHOLD {{ 2**2 }}



/********************************************************
   parameters
********************************************************/
#define MAX_SIZE_INPUTS_PER_LAYER {{ params.max_size_inputs_per_layer }}
#define MAX_SIZE_QINPUTS_PER_LAYER {{ params.max_size_qinputs_per_layer }}
#define MAX_SIZE_KN2ROW_BUFFER_PER_LAYER {{ params.max_size_kn2row_buffer_per_layer }}
#define MAX_SIZE_KN2ROW_COL_BLOCK {{ params.max_size_kn2row_col_block }}

#define MAX_SIZE_KERNELS_PER_LAYER {{ params.max_size_kernels_per_layer }}
#define MAX_SIZE_QKERNELS_PER_LAYER {{ params.max_size_qkernels_per_layer }}
#define MAX_SIZE_QKERNELS_PER_PE {{ params.max_size_qkernels_per_pe }}

#define MAX_SIZE_OUTPUTS_PER_LAYER {{ params.max_size_outputs_per_layer }}
#define MAX_SIZE_QOUTPUTS_PER_LAYER {{ params.max_size_qoutputs_per_layer }}

#define MAX_NBIT_QINPUT 2 // {{ params.max_nbit_qinput }}
#define MAX_NBIT_KERNEL 1 // {{ params.max_nbit_qkernel }}
#define MAX_IN_C 1024
/********************************************************/

void write_to_file(const char *filename, int id, volatile int32_t* data, int size);
void write_to_file(const char *filename, int id, BIN_CONV_OUTPUT* data, int size);
void write_to_file(const char *filename, int id, QUANTIZED_NOT_PACKED* data, int size);
void write_to_file(const char *filename, int id, float* data, int size);

#endif


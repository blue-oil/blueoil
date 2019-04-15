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
#include <type_traits>
#include "func/impl/pop_count.h"

typedef uint32_t T_UINT;
typedef int32_t  T_INT;
typedef float    T_FLOAT;
typedef uint8_t  T_UINT8;
typedef int8_t   T_INT8;
typedef uint16_t  T_UINT16;
typedef int16_t   T_INT16;

#define QUANTIZED_NOT_PACKED uint8_t

template <typename pack_type>
class QuantizedPacked {
 public:
  using T = pack_type;
  using base_t = std::remove_cv_t<T>;
  static constexpr std::size_t BitCount = sizeof(pack_type) * CHAR_BIT;
  QuantizedPacked() = default;
  explicit QuantizedPacked(const T val) : val(val) {}
  explicit operator base_t() const { return val; }
  template <typename U, std::enable_if_t<std::is_same<base_t, std::remove_cv_t<U>>::value, int> = 0>
  QuantizedPacked<T>& operator|=(const QuantizedPacked<U>& that) {
    val |= that.val;
    return *this;
  }
  base_t Raw() const { return val; }
 private:
  T val;
} __attribute__ ((packed));

#if defined RUN_ON_FPGA
  using QUANTIZED_PACKED = QuantizedPacked<volatile {{ params.default_qword_dtype.cpptype() }}>;
#else
  using QUANTIZED_PACKED = QuantizedPacked<{{ params.default_qword_dtype.cpptype() }}>;
#endif
using QUANTIZED_PACKED_KERNEL = QuantizedPacked<{{ params.default_qword_dtype.cpptype() }}>;
template <typename T1, typename T2,
    std::enable_if_t<std::is_same<std::remove_cv_t<T1>, std::remove_cv_t<T2>>::value, int> = 0>
inline auto operator^(const QuantizedPacked<T1>& lhs, const QuantizedPacked<T2>& rhs) {
  using packed_t = QuantizedPacked<std::remove_cv_t<T1>>;
  return packed_t(lhs.Raw() ^ rhs.Raw());
}
template <typename pack_type>
inline auto operator~(const QuantizedPacked<pack_type>& x) {
  return QuantizedPacked<std::remove_cv_t<pack_type>>(~x.Raw());
}
template <typename pack_type>
inline int pop_count(const QuantizedPacked<pack_type>& x) {
  return dlk::impl::pop_count(x.Raw());
}

template <typename T>
struct Base {
  using type = T;
};

template <typename T>
struct Base<QuantizedPacked<T>> {
  using type = T;
};

#if defined RUN_ON_FPGA
  typedef volatile T_INT16 BIN_CONV_OUTPUT;
#else
  typedef T_INT16 BIN_CONV_OUTPUT;
#endif

#define NBIT_QDYPE {{ params.default_nbit_qword }}

#define DEFAULT_GRAPH_INPUT {{ graph_input.dtype.cpptype() }}
#define DEFAULT_GRAPH_OUTPUT {{ graph_output.dtype.cpptype() }}

#define BIN_CONV_FORMULA_SCALING_FACTOR 3.0
#define WORD_SIZE 32

#define IP_CSR_ADDR 0xFF200000
#define TH_IP_CSR_ADDR 0xFF200100
#define INPUT_ADDR 0x20000000
#define OUTPUT0_ADDR 0x28000000
#define OUTPUT1_ADDR 0x30000000
#define OUTPUT_ADDR OUTPUT0_ADDR
#define KERNEL_ADDR 0x38000000
#define THRESHOLD_ADDR 0x3F000000

#define NUM_PE {{ config.num_pe }}

{%- if config.activate_hard_quantization %}
#define HARD_QUANTIZATION_ACTIVE
{% endif %}

{%- if config.threshold_skipping %}
#define THRESHOLD_SKIPPING_ACTIVE
{% endif %}

#define NUM_OF_A2W1_THRESHOLD {{ 2**2 }}

#define PS_PL_BANDWIDTH {{ config.bandwidth }}


/********************************************************
   parameters
********************************************************/
#define MAX_SIZE_INPUTS_PER_LAYER {{ params.max_size_inputs_per_layer }}
#define MAX_SIZE_QINPUTS_PER_LAYER {{ params.max_size_qinputs_per_layer }}
#define MAX_SIZE_IM2COL_INPUTS_PER_LAYER {{ params.max_size_im2col_inputs_per_layer }}
#define MAX_SIZE_IM2COL_QINPUTS_PER_LAYER {{ params.max_size_im2col_qinputs_per_layer }}
#define MAX_SIZE_KN2ROW_BUFFER_PER_LAYER {{ params.max_size_kn2row_buffer_per_layer }}

#define MAX_SIZE_KERNELS_PER_LAYER {{ params.max_size_kernels_per_layer }}
#define MAX_SIZE_QKERNELS_PER_LAYER {{ params.max_size_qkernels_per_layer }}
#define MAX_SIZE_QKERNELS_PER_PE {{ params.max_size_qkernels_per_pe }}

#define MAX_SIZE_OUTPUTS_PER_LAYER {{ params.max_size_outputs_per_layer }}
#define MAX_SIZE_QOUTPUTS_PER_LAYER {{ params.max_size_qoutputs_per_layer }}

#define MAX_NBIT_QINPUT 2 // {{ params.max_nbit_qinput }}
#define MAX_NBIT_KERNEL 1 // {{ params.max_nbit_qkernel }}
#define MAX_IN_C 1024
/********************************************************/

typedef T_INT Quantized_t;


#endif


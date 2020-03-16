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

#include <iostream>
#include <climits>
#include <cstring>
#include <cstdio>
#include <ctime>
#include "global.h"
#include "func/add.h"
#include "func/average_pool.h"
#include "func/batch_normalization.h"
#include "func/concat_on_depth.h"
#include "func/conv2d.h"
#include "func/depth_to_space.h"
#include "func/resize_nearest_neighbor.h"
#include "func/extract_image_patches.h"
#include "func/max.h"
#include "func/max_pool.h"
#include "func/minimum.h"
#include "func/pad.h"
#include "func/mul.h"
#include "func/matmul.h"
#include "func/quantized_conv2d.h"
#include "func/real_div.h"
#include "func/relu.h"
#include "func/leaky_relu.h"
#include "func/round.h"
#include "func/softmax.h"
#include "func/split.h"
#include "func/sqrt.h"
#include "func/sub.h"
#include "func/lookup.h"
#include "matrix/multiplication.h"
#include "operators.h"
#include "quantizer.h"
#include "network.h"

#ifdef HARD_QUANTIZATION_ACTIVE
#include "scaling_factors.h"
#endif

#ifdef THRESHOLD_SKIPPING_ACTIVE
#include "thresholds.h"
#endif

#include <chrono>
#include "operators.h"

#ifdef RUN_ON_FPGA
#include "memdriver.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

{% if config.debug -%}
#include "c2numpy.h"

void save_float32_data(const std::string &name, uint32_t size, uint32_t postfix, float *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), postfix, 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_FLOAT32);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_float32(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

void save_int32_data(const std::string &name, uint32_t size, uint32_t postfix, int32_t *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), postfix, 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_INT32);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_int32(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

void save_int16_data(const std::string &name, uint32_t size, uint32_t postfix, int16_t *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), postfix, 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_INT16);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_int16(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

void save_uint32_data(const std::string &name, uint32_t size, uint32_t postfix, uint32_t *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), postfix, 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_UINT32);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_uint32(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

{% endif %}
{{ '\n' -}}

/////////////////////////////////////////
// import inputs
/////////////////////////////////////////
{% for input in graph.consts -%}
#include "inputs/{{ input.name }}.h"
{% endfor %}
{{ '\n' -}}
/////////////////////////////////////////

Network::Network()
{}

Network::~Network()
{
  {% for node in graph.non_variables -%}
  {% if node.available_buffer == '' -%}
  {% for out_k in node.output_ops.keys() -%}
  delete []{{ node.name + '_' + out_k }}_raw;
  {%- endfor %}
  {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
  {% for out_k in node.output_ops.keys() -%}
  {% if out_k != node.output_ops.keys()|list|first %}
  delete []{{ node.name + '_' + out_k }}_raw;
  {% endif %}
  {%- endfor %}
  {% endif %}
  {%- endfor %}
  {{ '\n' -}}

#if defined RUN_ON_FPGA
#else
  delete [] device_input_buf;
  delete [] device_output_buf;
#endif
}

bool Network::init()
{

#if defined RUN_ON_FPGA

  if(
    !dma_input_buffer.init(
        {% if config.cache %}
          "udmabuf0",
        {% else %}
          "mem",
        {% endif %}
        max_device_input_elems,
        sizeof(QUANTIZED_PACKED),
        {% if config.cache %}
          true, 0
        {% else %}
          false, INPUT_ADDR
        {% endif %}
    )
  )
  {
    return false;
  }

  if(
    !dma_output_buffer.init(
        {% if config.cache %}
          "udmabuf1",
        {% else %}
          "mem",
        {% endif %}
        max_device_output_elems,
        sizeof(BIN_CONV_OUTPUT),
        {% if config.cache %}
          true, 0
        {% else %}
          false, OUTPUT_ADDR
        {% endif %}
    )
  )
  {
    return false;
  }

  device_input_buf = (QUANTIZED_PACKED*) dma_input_buffer.buffer();
  device_output_buf = (BIN_CONV_OUTPUT*) dma_output_buffer.buffer();

#else
  device_input_buf = new QUANTIZED_PACKED[max_device_input_elems]();
  device_output_buf = new BIN_CONV_OUTPUT[max_device_output_elems]();
#endif

#if !defined(RUN_ON_FPGA) && !defined(USE_NEON) && !defined(USE_AVX)
  qconv_tmp_buffer = std::make_unique<BYTE[]>(std::max({
      MAX_SIZE_KN2ROW_BUFFER_PER_LAYER * sizeof(BIN_CONV_OUTPUT),
      MAX_SIZE_QOUTPUTS_PER_LAYER * sizeof(QUANTIZED_PACKED),
      MAX_SIZE_OUTPUTS_PER_LAYER * sizeof(BIN_CONV_OUTPUT)
  }));
#endif
#ifdef _OPENMP
  const std::size_t thread_num = omp_get_max_threads();
#else
  const std::size_t thread_num = 1;
#endif
  conv_tmp_buffer = std::make_unique<BYTE[]>(
      MAX_SIZE_KERNELS_PER_LAYER * sizeof(float)
      + MAX_SIZE_KN2ROW_BUFFER_PER_LAYER * sizeof(float)
      + MAX_IN_C * MAX_SIZE_KN2ROW_COL_BLOCK * sizeof(float)
      + thread_num * MAX_IN_C * dlk::details::MAX_UNROLL * sizeof(float)
  );
  quantize_tmp_buffer = std::make_unique<BYTE[]>(
      MAX_SIZE_INPUTS_PER_LAYER * sizeof(QUANTIZED_NOT_PACKED)
  );

  {% for node in graph.non_variables -%}
  {% if node.available_buffer == '' %}
  {% for out_k in node.output_ops.keys() -%}
  {{ node.name + '_' + out_k }}_raw = new {{ node.dtype.cpptype() }}[{{ node.view.size_in_words_as_cpp }}]();
  {%- endfor %}
  {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
  {% for out_k in node.output_ops.keys() -%}
  {% if out_k != node.output_ops.keys()|list|first %}
  {{ node.name + '_' + out_k }}_raw = new {{ node.dtype.cpptype() }}[{{ node.view.size_in_words_as_cpp }}]();
  {% endif %}
  {%- endfor %}
  {% endif %}
  {%- endfor %}
  {{ '\n' -}}

#if defined RUN_ON_FPGA
  MappedMem kernel_mmap(KERNEL_ADDR, total_kernel_size);
  auto kernel_buffer = reinterpret_cast<uint8_t*>(kernel_mmap.get());
  {% for qconv in graph.convs(quantized_only=True) -%}
  {%    set kernel = qconv.input_nodes[1] -%}
  std::memcpy(kernel_buffer + {{qconv.name}}_kernel_offset, {{kernel.name}}_output.data(), {{qconv.name}}_kernel_size);
  {% endfor -%}

  MappedMem thresholds_mmap(THRESHOLD_ADDR, total_thresholds_size);
  auto thresholds_buffer = reinterpret_cast<uint8_t*>(thresholds_mmap.get());
  {% for qconv in graph.convs(quantized_only=True) -%}
  {% if qconv.has_thresholds -%}
  {% set thresholds = qconv.thresholds -%}
  std::memcpy(thresholds_buffer + {{qconv.name}}_thresholds_offset, const_cast<T_INT16*>({{qconv.name}}_thresholds), {{qconv.name}}_thresholds_size);
  {% endif -%}
  {% endfor -%}
#else
  {% for qconv in graph.convs(quantized_only=True) -%}
  {% if qconv.has_thresholds -%}
  dlk::impl::convert_thresholds({{ qconv.name }}_thresholds, {{ qconv.name }}_thresholds_converted.get(), {{ qconv.channel }});
  {% else -%}
  {% endif -%}
  {% endfor -%}
#endif // RUN_ON_FPGA

#pragma omp parallel
  std::cout << std::flush;

  return true;
}

int Network::get_input_rank()
{
  return input_rank;
}

int Network::get_output_rank()
{
  return output_rank;
}

void Network::get_input_shape(int32_t *shape)
{
  std::copy(input_shape, input_shape + input_rank, shape);
}

void Network::get_output_shape(int32_t *shape)
{
  std::copy(output_shape, output_shape + output_rank, shape);
}

bool Network::run(float *network_input, float *network_output)
{
  struct convolution_parameters Conv2D_struct;
  struct binary_convolution_parameters binConv2D_struct;
  struct max_pooling_parameters MaxPool_struct;
  struct avg_pooling_parameters AveragePool_struct;

  #if defined RUN_ON_FPGA
  binConv2D_struct.device_input_phys_addr = dma_input_buffer.physical_address();
  binConv2D_struct.device_output_phys_addr = dma_output_buffer.physical_address();

  binConv2D_struct.dma_input_buffer = &dma_input_buffer;
  binConv2D_struct.dma_output_buffer = &dma_output_buffer;
  #endif

  TensorView<{{ graph_input.dtype.cpptype() }}, MemoryLayout::{{ graph_input.dimension }}>::tensor_info_t<std::size_t> {{ graph_input.name }}_shape = {
    {% for len in graph_input.shape -%}
    {{- len -}},
    {%- endfor %}
  };
  TensorView<{{ graph_input.dtype.cpptype() }}, MemoryLayout::{{ graph_input.dimension }}> {{ graph_input.name }}_output(network_input, {{ graph_input.name }}_shape);
  {{ '\n' -}}

  {% for node in graph.non_variables -%}
  {% if node.available_buffer == '' %}
  {% for out_k in node.output_ops.keys() -%}
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>::tensor_info_t<std::size_t> {{ node.name + '_' + out_k }}_shape = {
    {% for len in node.shape -%}
    {{- len -}},
    {%- endfor %}
  };
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>
    {{ node.name + '_' + out_k }}({{ node.name + '_' + out_k }}_raw, {{ node.name + '_' + out_k }}_shape);
  {%- endfor %}
  {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
  {% for out_k in node.output_ops.keys() -%}
  {% if out_k != node.output_ops.keys()|list|first %}
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>::tensor_info_t<std::size_t> {{ node.name + '_' + out_k }}_shape = {
    {% for len in node.shape -%}
    {{- len -}},
    {%- endfor %}
  };
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}> {{ node.name + '_' + out_k }}({{ node.name + '_' + out_k }}_raw, {{ node.name + '_' + out_k }}_shape);
  {% endif %}
  {%- endfor %}
  {% endif %}
  {%- endfor %}
  {{ '\n' -}}

  {%- for node in graph.non_variables %}
  {{- node.view.run() }}

  {% if config.debug -%}
  {# Temporary: better access to the quantizer #}

  {% for out_k in node.output_ops.keys() -%}
  {% if node.dtype.cpptype() in ['int', 'int32_t'] -%}
  save_int32_data("debug/{{ node.name }}_{{ out_k }}", {{ node.view.size_in_words_as_cpp }}, 0, {{ node.name }}_{{ out_k }}.data(), 3.0 / 2.0 );
  {% elif node.dtype.cpptype() in ['unsigned', 'uint32_t'] -%}
  save_uint32_data("debug/{{ node.name }}_{{ out_k }}", {{ node.view.size_in_words_as_cpp }}, 0, {{ node.name }}_{{ out_k }}.data(), 1.0);
  {% elif node.dtype.cpptype() == 'QUANTIZED_PACKED' -%}
  save_uint32_data("debug/{{ node.name }}_{{ out_k }}", {{ node.view.size_in_words_as_cpp }}, 0, reinterpret_cast<uint32_t*>({{ node.name }}_{{ out_k }}.data()), 1.0);
  {% elif node.dtype.cpptype() == 'float' -%}
  save_float32_data("debug/{{ node.name }}_{{ out_k }}", {{ node.view.size_in_words_as_cpp }}, {{ loop.index0 }}, {{ node.name }}_{{ out_k }}.data(), 1.0);
  {{ '\n' -}}
  {% endif %}
  {%- endfor %}
  {% endif %}

  {% endfor -%}

  // TODO: support multiple output
  {% for out_k in graph_output.output_ops.keys() -%}
  std::copy({{ graph_output.name }}_{{ out_k }}.data(), {{ graph_output.name }}_{{ out_k }}.data() + {{ graph_output.view.size_in_words_as_cpp }}, network_output);
  {% endfor -%}

  return true;
}

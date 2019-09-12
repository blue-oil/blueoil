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
#include <vector>
#include <cstring>
#include <cstdio>
#include <ctime>
#include "global.h"
#include "func/add.h"
#include "func/average_pool.h"
#include "func/batch_normalization.h"
#include "func/concat_on_depth.h"
#include "func/concat_v2.h"
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
#include "func/quantize.h"
#include "func/quantized_conv2d.h"
#include "func/real_div.h"
#include "func/relu.h"
#include "func/leaky_relu.h"
#include "func/round.h"
#include "func/scale.h"
#include "func/softmax.h"
#include "func/split.h"
#include "func/sqrt.h"
#include "func/sub.h"
#include "func/unpooling.h"
#include "func/lookup.h"
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
#include <sys/mman.h>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

uint8_t* mapPhysicalMemory(size_t base, size_t size) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC, 0);
    if (fd == -1)
        throw std::system_error(errno, std::generic_category());
    int rw = PROT_READ | PROT_WRITE;
    auto* mapped_base = reinterpret_cast<uint8_t*>(mmap(nullptr, size, rw, MAP_SHARED, fd, base));
    if (mapped_base == MAP_FAILED)
        throw std::system_error(errno, std::generic_category());
    return mapped_base;
}

} // namespace

{% if config.debug -%}
#include "c2numpy.h"

void save_float32_data(const std::string &name, uint32_t size, float *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_FLOAT32);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_float32(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

void save_int32_data(const std::string &name, uint32_t size, int32_t *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_INT32);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_int32(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

void save_int16_data(const std::string &name, uint32_t size, int16_t *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_INT16);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_int16(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

void save_uint32_data(const std::string &name, uint32_t size, uint32_t *data, float scale)
{
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), 1<<31);
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
  {% if node.output_ops.keys()|length > 1 %}
  delete []{{ node.name + '_' + out_k }}_raw;
  {% else %}
  delete []{{ node.name }}_raw;
  {% endif %}
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

  {% for node in graph.non_variables -%}
  {% if node.available_buffer == '' %}
  {% for out_k in node.output_ops.keys() -%}
  {% if node.output_ops.keys()|length > 1 %}
  {{ node.name + '_' + out_k }}_raw = new {{ node.dtype.cpptype() }}[{{ node.view.shape }}]();
  {% else %}
  {{ node.name }}_raw = new {{ node.dtype.cpptype() }}[{{ node.view.shape }}]();
  {% endif %}
  {%- endfor %}
  {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
  {% for out_k in node.output_ops.keys() -%}
  {% if out_k != node.output_ops.keys()|list|first %}
  {{ node.name + '_' + out_k }}_raw = new {{ node.dtype.cpptype() }}[{{ node.view.shape }}]();
  {% endif %}
  {%- endfor %}
  {% endif %}
  {%- endfor %}
  {{ '\n' -}}

#if defined RUN_ON_FPGA
  auto* kernel_buffer = mapPhysicalMemory(KERNEL_ADDR, total_kernel_size);
  {% for qconv in graph.convs(quantized_only=True) -%}
  {%    set kernel = qconv.input_nodes[1] -%}
  std::memcpy(kernel_buffer + {{qconv.name}}_kernel_offset, {{kernel.name}}.data(), {{qconv.name}}_kernel_size);
  {% endfor -%}

  auto* thresholds_buffer = mapPhysicalMemory(THRESHOLD_ADDR, total_thresholds_size);
  {% for qconv in graph.convs(quantized_only=True) -%}
      {% if qconv.has_thresholds -%}
          {% set thresholds = qconv.thresholds -%}
  std::memcpy(thresholds_buffer + {{qconv.name}}_thresholds_offset, const_cast<T_INT16*>({{qconv.name}}_thresholds), {{qconv.name}}_thresholds_size);
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
  struct MaxPoolWithArgmax_parameters MaxPoolWithArgmax_struct;

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
  TensorView<{{ graph_input.dtype.cpptype() }}, MemoryLayout::{{ graph_input.dimension }}> {{ graph_input.name }}(network_input, {{ graph_input.name }}_shape);
  {{ '\n' -}}

  {% for node in graph.non_variables -%}
  {% if node.available_buffer == '' %}
  {% for out_k in node.output_ops.keys() -%}
  {% if node.output_ops.keys()|length > 1 %}
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>::tensor_info_t<std::size_t> {{ node.name + '_' + out_k }}_shape = {
    {% for len in node.shape -%}
    {{- len -}},
    {%- endfor %}
  };
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>
    {{ node.name + '_' + out_k }}({{ node.name + '_' + out_k }}_raw, {{ node.name + '_' + out_k }}_shape);
  {% else %}
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>::tensor_info_t<std::size_t> {{ node.name }}_shape = {
    {% for len in node.shape -%}
    {{- len -}},
    {%- endfor %}
  };
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>
    {{ node.name }}({{ node.name }}_raw, {{ node.name }}_shape);
  {% endif %}
  {%- endfor %}
  {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
  {% for out_k in node.output_ops.keys() -%}
  {% if out_k != node.output_ops.keys()|list|first %}
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>::tensor_info_t<std::size_t> {{ node.name + '_' + out_k }}_shape = {
    {% for len in node.shape -%}
    {{- len -}},
    {%- endfor %}
  };
  TensorView<{{ node.dtype.cpptype() }}, MemoryLayout::{{ node.dimension }}>
    {{ node.name + '_' + out_k }}({{ node.name + '_' + out_k }}_raw, {{ node.name + '_' + out_k }}_shape);
  {% endif %}
  {%- endfor %}
  {% endif %}
  {%- endfor %}
  {{ '\n' -}}

  {%- for node in graph.non_variables %}
  {{ node.view.run() }}

  {% if config.debug -%}
    {# Temporary: better access to the quantizer #}

    {% if node.dtype.cpptype() in ['int', 'int32_t'] -%}
      save_int32_data("debug/{{ node.name }}", {{ node.view.shape }}, {{ node.name }}, 3.0 / 2.0 );
    {% elif node.dtype.cpptype() in ['unsigned', 'uint32_t'] -%}
      save_uint32_data("debug/{{ node.name }}", {{ node.view.shape }}, {{ node.name }}, 1.0);
    {% elif node.dtype.cpptype() == 'float' -%}
      save_float32_data("debug/{{ node.name }}", {{ node.view.shape }}, {{ node.name }}, 1.0);
    {% endif %}
  {% endif %}

  {% endfor -%}

  std::copy({{ graph_output.name }}.data(), {{ graph_output.name }}.data() + {{ graph_output.view.shape }}, network_output);

  return true;
}

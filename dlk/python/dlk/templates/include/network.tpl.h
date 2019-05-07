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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "global.h"
#include "dma_buffer.h"

#define SYM_PUBLIC __attribute__ ((visibility ("default")))
#define SYM_LOCAL  __attribute__ ((visibility ("hidden")))

class SYM_PUBLIC Network
{
public:
    Network();
    ~Network();

    bool init();

    int get_input_rank();
    int get_output_rank();
    void get_input_shape(int32_t *shape);
    void get_output_shape(int32_t *shape);

    bool run(float *network_input, float *network_output);

private:
    // declarations
    {% for node in graph.non_variables %}
    {% if node.available_buffer == '' %}
    {% for out_k in node.output_ops.keys() -%}
    {% if node.output_ops.keys()|length > 1 %}
    {{ node.dtype.cpptype() }} *{{ node.name + '_' + out_k }} = 0;
    {% else %}
    {{ node.dtype.cpptype() }} *{{ node.name }} = 0;
    {% endif %}
    {%- endfor %}
    {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
    {% for out_k in node.output_ops.keys() -%}
    {% if out_k != node.output_ops.keys()|list|first %}
    {{ node.dtype.cpptype() }} *{{ node.name + '_' + out_k }} = 0;
    {% endif %}
    {%- endfor %}
    {% endif %}
    {%- endfor %}

    QUANTIZED_PACKED *device_input_buf = 0;
    BIN_CONV_OUTPUT *device_output_buf = 0;

    const T_INT input_rank = {{ graph_input.rank }};
    const T_INT input_shape[{{ graph_input.rank }}] = { {{ graph_input.view.shape_list }} };

    const T_INT output_rank = {{ graph_output.rank }};
    const T_INT output_shape[{{ graph_output.rank }}] = { {{ graph_output.view.shape_list }} };

    const int max_device_input_elems = MAX_SIZE_IM2COL_QINPUTS_PER_LAYER;
    const int max_device_output_elems = MAX_SIZE_OUTPUTS_PER_LAYER;

    DMA_Buffer dma_input_buffer;
    DMA_Buffer dma_output_buffer;

#if defined RUN_ON_FPGA
  {% set offset = namespace(o=0) -%}
  {% for qconv in graph.convs(quantized_only=True) -%}
  {%    set kernel = qconv.input_nodes[1] -%}
  {%    set n, h, w, c = kernel.shape -%}
  {%    set b = 32 -%}
  {%    set size = (((n + b - 1) // b) * b) * h * w * (((c + b - 1) // b) * b) // 32 * 4 -%}
  const uint32_t {{qconv.name}}_kernel_size = {{size}};
  const uint32_t {{qconv.name}}_kernel_offset = {{offset.o}};
  {%    set offset.o = offset.o + size -%}
  {% endfor -%}
  const uint32_t total_kernel_size = {{offset.o}};
#endif // RUN_ON_FPGA
};

#endif // NETWORK_H_INCLUDED


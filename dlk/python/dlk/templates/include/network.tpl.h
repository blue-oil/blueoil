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

#include <memory>
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
    {{ node.dtype.cpptype() }} *{{ node.name + '_' + out_k }}_raw = 0;
    {%- endfor %}
    {% elif node.available_buffer != '' and node.output_ops.keys()|length > 1 %}
    {% for out_k in node.output_ops.keys() -%}
    {% if out_k != node.output_ops.keys()|list|first %}
    {{ node.dtype.cpptype() }} *{{ node.name + '_' + out_k }}_raw = 0;
    {% endif %}
    {%- endfor %}
    {% endif %}
    {%- endfor %}

    QUANTIZED_PACKED *device_input_buf = 0;
    BIN_CONV_OUTPUT *device_output_buf = 0;

    std::unique_ptr<BYTE[]> qconv_tmp_buffer;
    std::unique_ptr<BYTE[]> conv_tmp_buffer;
    std::unique_ptr<BYTE[]> quantize_tmp_buffer;

    const T_INT input_rank = {{ graph_input.rank }};
    const T_INT input_shape[{{ graph_input.rank }}] = { {{ graph_input.view.shape_as_cpp }} };

    const T_INT output_rank = {{ graph_output.rank }};
    const T_INT output_shape[{{ graph_output.rank }}] = { {{ graph_output.view.shape_as_cpp }} };

    const int max_device_input_elems = MAX_SIZE_QINPUTS_PER_LAYER;
    const int max_device_output_elems = MAX_SIZE_OUTPUTS_PER_LAYER;

    DMA_Buffer dma_input_buffer;
    DMA_Buffer dma_output_buffer;

#if defined RUN_ON_FPGA
  {% set offset = namespace(o=0) -%}
  {% for qconv in graph.convs(quantized_only=True) -%}
  {%    set kernel = qconv.input_nodes[1] -%}
  {%    set oh, ih, kh, kw, ol, il = kernel.transposed_shape -%}
  {%    set b = 32 -%}
  {%    set size = oh * ih * kh * kw * ol * 32 // 8 -%}
  const uint32_t {{qconv.name}}_kernel_size = {{size}};
  const uint32_t {{qconv.name}}_kernel_offset = {{offset.o}};
  {%    set offset.o = offset.o + size -%}
  {% endfor -%}
  const uint32_t total_kernel_size = std::max(1, {{offset.o}});

  {% set th_offset = namespace(o=0) -%}
  {% for qconv in graph.convs(quantized_only=True) -%}
  {%     if qconv.has_thresholds -%}
  {%         set thresholds = qconv.thresholds -%}
  {%         set b = 32 -%}
  {%         set size = thresholds|length * b // 8 -%}
  const uint32_t {{qconv.name}}_thresholds_size = {{size}};
  const uint32_t {{qconv.name}}_thresholds_offset = {{th_offset.o}};
  {%         set th_offset.o = th_offset.o + size -%}
  {%     endif -%}
  {% endfor -%}
  const uint32_t total_thresholds_size = std::max(1, {{th_offset.o}});
#endif // RUN_ON_FPGA
  {% for qconv in graph.convs(quantized_only=True) -%}
  {%     if qconv.has_thresholds -%}
  {%         set b = 32 -%}
  {%         set channels_padded = qconv.channel + (b - qconv.channel % b) % b -%}
  const std::unique_ptr<BIN_CONV_OUTPUT[]> {{qconv.name}}_thresholds_converted = std::make_unique<BIN_CONV_OUTPUT[]>({{channels_padded}} * NUM_OF_A2W1_THRESHOLD);
  {%     else -%}
  const std::unique_ptr<BIN_CONV_OUTPUT[]> {{qconv.name}}_thresholds_converted;
  {%     endif -%}
  {% endfor -%}
};

#endif // NETWORK_H_INCLUDED


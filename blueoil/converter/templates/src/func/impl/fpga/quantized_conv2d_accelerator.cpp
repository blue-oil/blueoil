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
=============================================================================*/

#include <cassert>
#include <cstdio>

#include "de10_nano.h"
#include "func/impl/quantized_conv2d_accelerator.h"
#include "global.h"
#include "network.h"
#include "pack_input_to_qwords.h"
#include "time_measurement.h"

namespace
{
const unsigned int in_nbits = 2;
const unsigned int byte_nbits = 8;
} // namespace

namespace dlk
{

namespace impl
{

void TCAConv2d(const tca_input_t& input,
    const tca_kernel_t& kernel,
    const binary_convolution_parameters &p) {

  using namespace dlk;

  convolution_parameters cp = p.normal_conv_params;
  const T_UINT b = 32;
  const T_UINT out_c = ((cp.output_channels + b - 1) / b) * b;

  const T_UINT k_h = cp.kernel_height;
  const T_UINT k_w = cp.kernel_width;
  const T_UINT k_c = cp.kernel_depth;

  const T_UINT in_h = cp.input_height;
  const T_UINT in_w = cp.input_width;

  const T_UINT out_h = cp.output_height;
  const T_UINT out_w = cp.output_width;

  const auto effective_kernel_depth = ((cp.kernel_depth + b - 1) / b) * b;

    T_UINT input_byte_size =
        (cp.input_height * cp.input_width * effective_kernel_depth * in_nbits) /
        byte_nbits;

    T_UINT output_byte_size = out_h * out_w * out_c * sizeof(BIN_CONV_OUTPUT);
    if (p.thresholds != NULL) {
      output_byte_size /= 8;
    }

    Measurement::Start("Sync UDMABuf Input");
    p.dma_input_buffer->sync_size(input_byte_size);
    p.dma_input_buffer->sync_for_device();
    Measurement::Stop();

    Measurement::Start("Conv2D TCA");
    de10_nano::RunTCA(p.device_input_phys_addr, p.device_output_phys_addr, p.device_kernel_phys_addr, p.device_thresholds_phys_addr, in_w, in_h,
      k_c, out_w, out_h, out_c, k_w, k_h, cp.padding, cp.stride_along_height);
    Measurement::Stop();

    Measurement::Start("Sync UDMABuf Output");
    p.dma_output_buffer->sync_size(output_byte_size);
    p.dma_output_buffer->sync_for_cpu();
    Measurement::Stop();
}

} // namespace impl

} // namespace dlk

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

#include <cassert>

#include "func/impl/quantized_conv2d_dim2col.h"
#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "pack_input_to_qwords.h"
#include "time_measurement.h"

namespace dlk {

namespace impl {

void im2col(
  const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& input,
  const dlk::impl::dim2col_input_t& output,
  const binary_convolution_parameters& p)
{
  Measurement::Start("im2col");

  const auto n = p.normal_conv_params;

  int input_height = int(n.input_height);
  int input_width = int(n.input_width);

  unsigned ih_offset = n.input_width * n.kernel_depth;
  unsigned iw_offset = n.kernel_depth;
  unsigned in_padding = n.padding;

  QUANTIZED_NOT_PACKED tmp[QUANTIZED_PACKED::BitCount];
  std::size_t index = 0;
  for (unsigned oh = 0; oh < n.output_height; oh++) {
    for (unsigned ow = 0; ow < n.output_width; ow++) {
      for (unsigned kh = 0; kh < n.kernel_height; kh++) {
        for (unsigned kw = 0; kw < n.kernel_width; kw++) {
          for (unsigned kd = 0; kd < n.kernel_depth; kd++)
          {
            int ih = oh + kh - in_padding;
            int iw = ow + kw - in_padding;
            QUANTIZED_NOT_PACKED tmp_input = 0;

            if (ih < 0 || ih >= input_height)
              tmp_input = 0;
            else if (iw < 0 || iw >= input_width)
              tmp_input = 0;
            else
              tmp_input = input(0, ih, iw, kd);

            tmp[index++] = tmp_input;
            if (index == QUANTIZED_PACKED::BitCount) {
              for (unsigned bit_ch = 0; bit_ch < p.bin_input_bitwidth; ++bit_ch) {
                QUANTIZED_PACKED x(0);
                for (unsigned i = 0; i < QUANTIZED_PACKED::BitCount; ++i) {
                  x |= QUANTIZED_PACKED(((tmp[i] >> bit_ch) & QUANTIZED_PACKED::base_t(1)) << i);
                }
                output(oh * n.output_width + ow,
                    (kh * n.kernel_width * n.kernel_depth + kw * n.kernel_depth + kd) / QUANTIZED_PACKED::BitCount, 
                    bit_ch,
                    0) = x;
              }
              index = 0;
            }
          }
        }
      } // for (unsigned kh = 0; kh < kernel_height; kh++)
    }
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk

namespace {

int pop_count(T_UINT i) {
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void binary_convolution_cpu(const dlk::impl::dim2col_input_t& input_channels,
                            BIN_CONV_OUTPUT result[], const QUANTIZED_PACKED_KERNEL *kernel,
                            T_UINT output_channel_index,
                            struct binary_convolution_parameters bcp) {
  const T_UINT num_in_channels = bcp.bin_input_bitwidth;

  T_UINT bin_kernel_nwords = bcp.bin_kernel_ndata;
  T_UINT patches = bcp.normal_conv_params.output_height *
                   bcp.normal_conv_params.output_width;
  T_UINT output_depth = bcp.normal_conv_params.output_channels;

  const T_UINT remaining_oc = output_depth - output_channel_index;
  const unsigned num_kernels = (remaining_oc < NUM_PE) ? remaining_oc : NUM_PE;

  BIN_CONV_OUTPUT *thresholds;

  if (bcp.thresholds != nullptr) {
    thresholds = &bcp.thresholds[output_channel_index * NUM_OF_A2W1_THRESHOLD];
  } else {
    thresholds = nullptr;
  }

  for (T_UINT p = 0; p < patches; p++) {
    T_INT out[NUM_PE] = {};

    for (T_UINT idx_k = 0; idx_k < bin_kernel_nwords; idx_k++) {
      for (T_UINT in_channel = 0; in_channel < num_in_channels; in_channel++) {
        QUANTIZED_PACKED in_data = input_channels(p, idx_k, in_channel, 0);

        for (T_UINT k_pe = 0; k_pe < num_kernels; k_pe++) {
          const auto kernel_buf = kernel[k_pe * bin_kernel_nwords + idx_k];
          T_INT xnor_result = pop_count(~(in_data ^ kernel_buf));
          T_UINT kernel_bit_count = pop_count(~kernel_buf);

          T_INT conv_result;
          conv_result = (xnor_result - kernel_bit_count) << in_channel;

          out[k_pe] += conv_result;
        }
      }
    }

    for (T_UINT k_pe = 0; k_pe < num_kernels; k_pe++) {
      int idx_in = 0;
      int thresholds_offset = k_pe * NUM_OF_A2W1_THRESHOLD;

      T_INT conv_result = out[k_pe];
      T_INT output_buf;

      if (thresholds != nullptr) {
        T_INT ts0 = thresholds[thresholds_offset];
        T_INT ts1 = thresholds[thresholds_offset + 1];
        T_INT ts2 = thresholds[thresholds_offset + 2];
        auto flag = thresholds[thresholds_offset + 3]; // 1 for increasing, -1
                                                       // for decreasing, and
                                                       // constant otherwise

        if (flag == 1) // increasing function
        {
          if (conv_result < ts0)
            output_buf = 0;
          else if (conv_result < ts1)
            output_buf = 1;
          else if (conv_result < ts2)
            output_buf = 2;
          else
            output_buf = 3;
        } else if (flag == -1) // decreasing function
        {
          if (conv_result > ts2)
            output_buf = 0;
          else if (conv_result > ts1)
            output_buf = 1;
          else if (conv_result > ts0)
            output_buf = 2;
          else
            output_buf = 3;
        } else {                                      // constant function
          output_buf = flag - 2;                      // 2 is a magic number
          assert(0 <= output_buf && output_buf <= 3); // unsinged 2bits
        }
      } else {
        output_buf = conv_result;
      }

      int idx_out = p * output_depth + output_channel_index + k_pe;
      result[idx_out] = output_buf;
    }
  }
}

void binary_convolution(const dlk::impl::dim2col_input_t& input_channels,
                        BIN_CONV_OUTPUT result[],
                        QUANTIZED_PACKED_KERNEL kernel[MAX_SIZE_QKERNELS_PER_PE],
                        T_UINT output_channel_index,
                        struct binary_convolution_parameters bcp) {
  binary_convolution_cpu(input_channels, result, kernel, output_channel_index,
                         bcp);
}

} // namespace

namespace dlk {

namespace impl {

void QuantizedConv2DIm2Col(const dim2col_input_t& input, const kernel_t& kernel,
                                  const binary_convolution_parameters &p) {
  convolution_parameters cp = p.normal_conv_params;
  const T_UINT out_c = cp.output_channels;

  int ic = p.normal_conv_params.kernel_depth;
  int oh = p.normal_conv_params.output_height;
  int ow = p.normal_conv_params.output_width;
  int kh = p.normal_conv_params.kernel_height;
  int kw = p.normal_conv_params.kernel_width;

  Measurement::Start("QConv2D with im2col");
  for (T_UINT oc = 0; oc < out_c; oc += NUM_PE) {
    binary_convolution(input, p.device_output_buf,
                       kernel.data(oc, 0, 0, 0), oc, p);
  } // for (unsigned od = 0; od < output_depth; od++)
  Measurement::Stop();
}

} // namespace impl

} // namespace impl

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

#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "pack_input_to_qwords.h"
#include "time_measurement.h"

namespace {

template<typename T>
void im2col(
  QUANTIZED_NOT_PACKED input[],
  volatile T output[],
  struct convolution_parameters p)
{
  Measurement::Start("im2col");

  unsigned idx_out = 0;
  T tmp_input = 0;

  int input_height = int(p.input_height);
  int input_width = int(p.input_width);

  unsigned ih_offset = p.input_width * p.kernel_depth;
  unsigned iw_offset = p.kernel_depth;
  unsigned in_padding = p.padding;

  for (unsigned oh = 0; oh < p.output_height; oh++) {
    for (unsigned ow = 0; ow < p.output_width; ow++) {
      for (unsigned kh = 0; kh < p.kernel_height; kh++) {
        for (unsigned kw = 0; kw < p.kernel_width; kw++) {
          for (unsigned kd = 0; kd < p.kernel_depth; kd++)
          {
            int ih = oh + kh - in_padding;
            int iw = ow + kw - in_padding;

            if (ih < 0 || ih >= input_height)
              tmp_input = 0;
            else if (iw < 0 || iw >= input_width)
              tmp_input = 0;
            else
              tmp_input = input[ih * ih_offset + iw * iw_offset + kd];

            output[idx_out++] = tmp_input;
          }
        }
      } // for (unsigned kh = 0; kh < kernel_height; kh++)
    }
  }

  Measurement::Stop();
}

int pop_count(T_UINT i) {
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void binary_convolution_cpu(QUANTIZED_PACKED input_channels[],
                            BIN_CONV_OUTPUT result[], QUANTIZED_PACKED_KERNEL *kernel,
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

  unsigned idx_in = 0;

  for (T_UINT p = 0; p < patches; p++) {
    T_INT out[NUM_PE] = {};

    for (T_UINT idx_k = 0; idx_k < bin_kernel_nwords; idx_k++) {
      for (T_UINT in_channel = 0; in_channel < num_in_channels; in_channel++) {
        QUANTIZED_PACKED in_data = input_channels[idx_in];

        for (T_UINT k_pe = 0; k_pe < num_kernels; k_pe++) {
          const auto kernel_buf = kernel[k_pe * bin_kernel_nwords + idx_k];
          T_INT xnor_result = pop_count(~(in_data ^ kernel_buf));
          T_UINT kernel_bit_count = pop_count(~kernel_buf);

          T_INT conv_result;
          conv_result = (xnor_result - kernel_bit_count) << in_channel;

          out[k_pe] += conv_result;
        }

        idx_in++;
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

void binary_convolution(QUANTIZED_PACKED input_channels[],
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

void QuantizedConv2DIm2Col(QUANTIZED_NOT_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                                  const binary_convolution_parameters &p) {
  convolution_parameters cp = p.normal_conv_params;
  static QUANTIZED_NOT_PACKED
      im2col_input_buf[MAX_SIZE_IM2COL_INPUTS_PER_LAYER] = {};
  const T_UINT out_c = cp.output_channels;

  Measurement::Start("Im2col");
  im2col(input, im2col_input_buf, p.normal_conv_params);
  Measurement::Stop();

  Measurement::Start("Packing input for im2col");
  int ic = p.normal_conv_params.kernel_depth;
  int oh = p.normal_conv_params.output_height;
  int ow = p.normal_conv_params.output_width;
  int kh = p.normal_conv_params.kernel_height;
  int kw = p.normal_conv_params.kernel_width;

  unsigned im2col_input_elems = oh * ow * kh * kw * ic;
  // pack_input_to_qwords(im2col_input_buf, p.device_input_buf, p);
  pack_input_to_qwords(im2col_input_buf, p.device_input_buf, im2col_input_elems,
                       2);
  Measurement::Stop();

  Measurement::Start("QConv2D with im2col");
  for (T_UINT oc = 0; oc < out_c; oc += NUM_PE) {
    binary_convolution(p.device_input_buf, p.device_output_buf,
                       &kernel[oc * p.bin_kernel_ndata], oc, p);
  } // for (unsigned od = 0; od < output_depth; od++)
  Measurement::Stop();
}

} // namespace impl

} // namespace impl

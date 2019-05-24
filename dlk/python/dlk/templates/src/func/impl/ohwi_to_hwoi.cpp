/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

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

#include "func/impl/quantized_conv2d_kn2row.h"
#include "time_measurement.h"

namespace dlk {

namespace impl {

// kernel format converter
// ohwi : oc kh kw ic, hwoi: kh kw oc ic
void quantized_ohwi_to_hwoi(const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::NHWC>& ohwi, const kn2row_kernel_t& hwoi, const binary_convolution_parameters& p) {
  Measurement::Start("quantized_ohwi_to_hwoi");

  int ic = p.normal_conv_params.kernel_depth / 32;
  int oc = p.normal_conv_params.output_channels;
  int kh = p.normal_conv_params.kernel_height;
  int kw = p.normal_conv_params.kernel_width;

  for (unsigned int i = 0; i < kh; ++i) {
    for (unsigned int j = 0; j < kw; ++j) {
      for (unsigned int k = 0; k < oc; ++k) {
        for (unsigned int l = 0; l < ic; ++l) {
          hwoi(i, j, k, l) = ohwi(k, i, j, l);
        }
      }
    }
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk


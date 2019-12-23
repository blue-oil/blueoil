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
=============================================================================*/

#include "test_util.hpp"
#include "blueoil.hpp"


namespace blueoil {
namespace util {

Tensor Tensor_CHW_to_HWC(const Tensor &tensor) {
    auto shape = tensor.shape();
    const int channels  = shape[0];
    const int height = shape[1];
    const int width  = shape[2];
    Tensor dstTensor({height, width, channels});
    int srcPlaneSize = width * height;
    const float *srcImagePtr = tensor.dataAsArray();
    float *dstImagePtr = dstTensor.dataAsArray();
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const float *srcPixelPtr0 = srcImagePtr + x + (y * height);
        for (int c = 0; c < channels; c++) {
          const float *srcPixelPtr = srcPixelPtr0 + (c * srcPlaneSize);
          *dstImagePtr = *srcPixelPtr;
          dstImagePtr++;
        }
      }
    }
    return dstTensor;
}

Tensor Tensor_HWC_to_CHW(const Tensor &tensor) {
    auto shape = tensor.shape();
    int height = shape[0];
    int width  = shape[1];
    int channels = shape[2];
    Tensor dstTensor({channels, height, width});
    const float *srcImagePtr = tensor.dataAsArray();
    float *dstImagePtr = dstTensor.dataAsArray();
    for (int c = 0; c < channels; c++) {
      const float *srcPixelPtr = srcImagePtr + c;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          *dstImagePtr = *srcPixelPtr;
          srcPixelPtr += channels;
          dstImagePtr++;
        }
      }
    }
    return dstTensor;
}

}  // namespace util
}  // namespace blueoil

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

#include <iostream>
#include <cmath>
#include <cassert>
#include <string>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_npy.hpp"
#ifdef USE_OPENCV
#include "blueoil_opencv.hpp"
#endif
#ifdef USE_LIBPNG
#include "blueoil_png.hpp"
#endif

namespace blueoil {
namespace image {


template <typename T>
T clamp(const T x, const T lowerLimit, const T upperLimit) {
  if (x < lowerLimit) {
    return lowerLimit;
  }
  if (upperLimit < x) {
    return upperLimit;
  }
  return x;
}


Tensor LoadImage(const std::string filename) {
  blueoil::Tensor tensor({0});
#ifdef USE_OPENCV
  cv::Mat img = cv::imread(filename, 1);  // 1:force to RGB format
  if (!img.empty()) {
    return blueoil::opencv::Tensor_fromCVMat(img);
  }
#elif USE_LIBPNG
  tensor = blueoil::png::Tensor_fromPNGFile(filename);
  if (tensor.shape()[0] > 0) {
    return tensor;
  }
#endif
  tensor = blueoil::npy::Tensor_fromNPYFile(filename);
  if (tensor.shape().size() != 3) {
    throw std::runtime_error("npy image shape must be 3-dimention");
  }
  int channels = tensor.shape()[2];
  if ((channels != 1) && (channels != 3)) {
    throw std::runtime_error("npy image channels must be 1(grayscale) or 3(RGB)");
  }
  return tensor;
}

/*
 * Resize Image (Nearest Neighbor)
 */
Tensor ResizeHorizontal_NearestNeighbor(const Tensor &tensor, const int width) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int height = srcHeight;
  Tensor dstTensor({height, width, channels});
  //
  int srcRGBlinesize = srcWidth * channels;
  float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
  float srcRGBscaled = 1.0f / xScale;
  const float *srcImageData = tensor.dataAsArray();
  float *srcRGBline = const_cast<float *>(srcImageData);
  float *dstRGB = dstTensor.dataAsArray();
  for (int dstY = 0 ; dstY < height ; dstY++) {
    float srcRGBindexF = 0.5 / xScale;
    for (int dstX = 0 ; dstX < width ; dstX++) {
      float *srcRGB = srcRGBline + (static_cast<int>(srcRGBindexF) * channels);
      for (int c = 0 ; c < channels ; c++) {
        *dstRGB++ = *srcRGB++;
      }
      srcRGBindexF += srcRGBscaled;
    }
    srcRGBline += srcRGBlinesize;
  }
  return dstTensor;
}

Tensor ResizeVertical_NearestNeighbor(const Tensor &tensor, const int height) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int width = srcWidth;
  Tensor dstTensor({height, width, channels});
  const int srcScanLineSize = width * channels;
  float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
  float srcRGBscaled = 1.0f / yScale;
  const float *srcImageData = tensor.dataAsArray();
  float *srcRGBbase = const_cast<float *>(srcImageData);
  float *dstRGB = dstTensor.dataAsArray();
  float srcRGBindexF = 0.5 / yScale;
  for (int dstY = 0 ; dstY < height ; dstY++) {
    float *srcRGB = srcRGBbase + (static_cast<int>(srcRGBindexF) * srcScanLineSize);
    for (int i = 0 ; i < srcScanLineSize ; i++) {
      *dstRGB++ = *srcRGB++;
    }
    srcRGBindexF += srcRGBscaled;
  }
  return dstTensor;
}

/*
 * Resize Image (Bi-Linear)
 */
Tensor ResizeHorizontal_BiLinear(const Tensor &tensor, const int width) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int height = srcHeight;
  Tensor dstTensor({height, width, channels});
  float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
  int xSrcWindow = std::floor(1/xScale);
  xSrcWindow = (xSrcWindow < 2)? 2 :xSrcWindow;
  for (int dstY = 0 ; dstY < height ; dstY++) {
    for (int dstX = 0 ; dstX < width ; dstX++) {
      int srcX = static_cast<int>(std::floor(dstX/xScale));
      int srcY = dstY;
      for (int c = 0 ; c < channels ; c++) {
        float v = 0.0;
        float totalW = 0.0;
        for (int x = -xSrcWindow ; x < xSrcWindow; x++) {
          int srcX2 = clamp(srcX + x, 0, srcWidth - 1);
          const float *srcRGB = tensor.dataAsArray({srcY, srcX2, 0});
          float d = std::abs(static_cast<float>(x) / static_cast<float> (xSrcWindow));
          float w = 1.0 - d;  // Bi-Linear
          v += w * srcRGB[c];
          totalW += w;
        }
        float *dstRGB = dstTensor.dataAsArray({dstY, dstX, 0});
        dstRGB[c] = v / totalW;
      }
    }
  }
  return dstTensor;
}

Tensor ResizeVertical_BiLinear(const Tensor &tensor, const int height) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int width = srcWidth;
  Tensor dstTensor({height, width, channels});
  float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
  int ySrcWindow = std::floor(1/yScale);
  ySrcWindow = (ySrcWindow < 2)? 2 :ySrcWindow;
  for (int dstY = 0 ; dstY < height ; dstY++) {
    for (int dstX = 0 ; dstX < width ; dstX++) {
      int srcX = dstX;
      int srcY = static_cast<int>(std::floor(dstY/yScale));
      for (int c = 0 ; c < channels ; c++) {
        float v = 0.0;
        float totalW = 0.0;
        for (int y = -ySrcWindow ; y < ySrcWindow ; y++) {
          int srcY2 = clamp(srcY + y, 0, srcHeight - 1);
          const float *srcRGB = tensor.dataAsArray({srcY2, srcX, 0});
          float d = std::abs(static_cast<float>(y) / static_cast<float> (ySrcWindow));
          float w = 1.0 - d;  // Bi-Linear
          v += w * srcRGB[c];
          totalW += w;
        }
        float *dstRGB = dstTensor.dataAsArray({dstY, dstX, 0});
        dstRGB[c] = v / totalW;
      }
    }
  }
  return dstTensor;
}

Tensor Resize(const Tensor& image, const int width, const int height,
              const enum ResizeFilter filter) {
  auto shape = image.shape();
  int channels = shape[2];
  assert(shape.size() == 3);  // 3D shape: HWC
  assert((channels == 1) || (channels == 3));  // grayscale or RGB
  assert((filter == RESIZE_FILTER_NEAREST_NEIGHBOR) || (filter == RESIZE_FILTER_BI_LINEAR));
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  Tensor dstImage = image;
  if  (srcWidth != width) {
    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
      dstImage = ResizeHorizontal_NearestNeighbor(dstImage, width);
    } else {  // RESIZE_FILTER_BI_LINEAR
      dstImage = ResizeHorizontal_BiLinear(dstImage, width);
    }
  }
  if  (srcHeight != height) {
    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
      dstImage = ResizeVertical_NearestNeighbor(dstImage, height);
    } else {  // RESIZE_FILTER_BI_LINEAR
      dstImage = ResizeVertical_BiLinear(dstImage, height);
    }
  }
  return dstImage;
}


}  // namespace image
}  // namespace blueoil

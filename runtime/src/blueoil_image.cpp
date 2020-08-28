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
  for (int dstY = 0; dstY < height; dstY++) {
    float srcRGBindexF = 0.5 / xScale;
    for (int dstX = 0; dstX < width; dstX++) {
      float *srcRGB = srcRGBline + (static_cast<int>(srcRGBindexF) * channels);
      for (int c = 0; c < channels; c++) {
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
  for (int dstY = 0; dstY < height; dstY++) {
    float *srcRGB = srcRGBbase + (static_cast<int>(srcRGBindexF) * srcScanLineSize);
    for (int i = 0; i < srcScanLineSize; i++) {
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
  const float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
  const float xSrcWindow = (xScale < 1.0)? (1.0f/xScale): 1.0;
  for (int dstY = 0; dstY < height; dstY++) {
    for (int dstX = 0; dstX < width; dstX++) {
      float srcXf = (dstX + 0.5)/xScale - 0.5;
      int srcY = dstY;
      int xStart = std::ceil(srcXf - xSrcWindow);
      int xEnd = std::floor(srcXf + xSrcWindow);
      if (xStart >= xEnd) {  // for enlarge scale
        xStart = std::floor(srcXf);
        xEnd = std::ceil(srcXf);
      }
      // don't convolve pixels outside the frame
      if (xStart < 0) {
        xStart = 0;
      }
      if (xEnd >= srcWidth) {
        xEnd = srcWidth - 1;
      }
      for (int c = 0; c < channels; c++) {
        float v = 0.0;
        float totalW = 0.0;
        for (int srcX = xStart; srcX <= xEnd; srcX++) {
          float d = std::abs(static_cast<float>(srcX) - srcXf) / xSrcWindow;
          if (d < 1.0) {
            const float *srcPixel = tensor.dataAsArray({srcY, srcX, c});
            float w = 1.0 - d;  // Bi-Linear
            v += w * (*srcPixel);
            totalW += w;
          }
        }
        float *dstPixel = dstTensor.dataAsArray({dstY, dstX, c});
        *dstPixel = (v)? (v/totalW): 0;
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
  const float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
  const int ySrcWindow = (yScale < 1.0)? (1.0f/yScale): 1.0;
  for (int dstX = 0; dstX < width; dstX++) {
    for (int dstY = 0; dstY < height; dstY++) {
      int srcX = dstX;
      float srcYf = (dstY + 0.5)/yScale - 0.5;
      int yStart = std::ceil(srcYf - ySrcWindow);
      int yEnd = std::floor(srcYf + ySrcWindow);
      if (yStart >= yEnd) {  // for enlarge scale
        yStart = std::floor(srcYf);
        yEnd = std::ceil(srcYf);
      }
      // don't convolve pixels outside the frame
      if (yStart < 0) {
        yStart = 0;
      }
      if (yEnd >= srcHeight) {
        yEnd = srcHeight - 1;
      }
      for (int c = 0; c < channels; c++) {
        float v = 0.0;
        float totalW = 0.0;
        for (int srcY = yStart; srcY <= yEnd; srcY++) {
          float d = std::abs(static_cast<float>(srcY) - srcYf) / ySrcWindow;
          if (d < 1.0) {
            const float *srcPixel = tensor.dataAsArray({srcY, srcX, c});
            float w = 1.0 - d;  // Bi-Linear
            v += w * (*srcPixel);
            totalW += w;
          }
        }
        float *dstPixel = dstTensor.dataAsArray({dstY, dstX, c});
        *dstPixel = (v)? (v/totalW): 0;
      }
    }
  }
  return dstTensor;
}

Tensor Resize(const Tensor& image, const int height, const int width,
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

/* Copyright 2018 Leapmind Inc. */

#include <iostream>
#include <cmath>
#include <cassert>

#include "blueoil.hpp"
#include "blueoil_image.hpp"

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


Tensor Tensor_CHW_to_HWC(Tensor &tensor) {
  auto shape = tensor.shape();
  const int channels  = shape[0];
  const int height = shape[1];
  const int width  = shape[2];
  Tensor dstTensor({height, width, channels});
  int srcPlaneSize = width * height;
  float *srcImagePtr = tensor.dataAsArray();
  float *dstImagePtr = dstTensor.dataAsArray();
  for (int y = 0 ; y < height ; y++) {
    for (int x = 0 ; x < width ; x++) {
      float *srcPixelPtr0 = srcImagePtr + x + (y * height);
      for (int c = 0 ; c < channels ; c++) {
        float *srcPixelPtr = srcPixelPtr0 + (c * srcPlaneSize);
	*dstImagePtr = *srcPixelPtr;
	dstImagePtr++;
      }
    }
  }
  return dstTensor;
}

Tensor Tensor_HWC_to_CHW(Tensor &tensor) {
  auto shape = tensor.shape();
  int height = shape[0];
  int width  = shape[1];
  int channels = shape[2];
  Tensor dstTensor({channels, height, width});
  float *srcImagePtr = tensor.dataAsArray();
  float *dstImagePtr = dstTensor.dataAsArray();
  for (int c = 0 ; c < channels ; c++) {
    float *srcPixelPtr = srcImagePtr + c;
    for (int y = 0 ; y < height ; y++) {
      for (int x = 0 ; x < width ; x++) {
	*dstImagePtr = *srcPixelPtr;
	srcPixelPtr += channels;
	dstImagePtr++;
      }
    }
  }
  return dstTensor;
}


/*
 * Resize Image (Nearest Neighbor)
 */
Tensor ResizeHorizontal_NearestNeighbor(Tensor &tensor, const int width) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int height = srcHeight;
  Tensor dstTensor({height, width, channels});
  float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
  for (int dstY = 0 ; dstY < height ; dstY++) {
    for (int dstX = 0 ; dstX < width ; dstX++) {
      int srcX = (int) std::floor(dstX/xScale);
      int srcY = dstY;
      for (int c = 0 ; c < channels ; c++) {
	float *srcRGB = tensor.dataAsArray({srcY, srcX, 0});
	float *dstRGB = dstTensor.dataAsArray({dstY, dstX, 0});
	dstRGB[c] = srcRGB[c];
      }
    }
  }
  return dstTensor;
}

Tensor ResizeVertical_NearestNeighbor(Tensor &tensor, const int height) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int width = srcWidth;
  Tensor dstTensor({height, width, channels});
  float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
  for (int dstY = 0 ; dstY < height ; dstY++) {
    for (int dstX = 0 ; dstX < width ; dstX++) {
      int srcX = dstX;
      int srcY = (int) std::floor(dstY/yScale);
      for (int c = 0 ; c < channels ; c++) {
	float *srcRGB = tensor.dataAsArray({srcY, srcX, 0});
	float *dstRGB = dstTensor.dataAsArray({dstY, dstX, 0});
	dstRGB[c] = srcRGB[c];
      }
    }
  }
  return dstTensor;
}

/*
 * Resize Image (Bi-Linear)
 */
Tensor ResizeHorizontal_BiLinear(Tensor &tensor, const int width) {
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
      int srcX = (int) std::floor(dstX/xScale);
      int srcY = dstY;
      for (int c = 0 ; c < channels ; c++) {
	float v = 0.0;
	float totalW = 0.0;
	for (int x = -xSrcWindow ; x < xSrcWindow; x++){
	  int srcX2 = clamp(srcX + x, 0, srcWidth - 1);
	  float *srcRGB = tensor.dataAsArray({srcY, srcX2, 0});
	  float d = std::abs(static_cast<float>(x) / static_cast<float> (xSrcWindow));
	  float w = 1.0 - d; // Bi-Linear
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

Tensor ResizeVertical_BiLinear(Tensor &tensor, const int height) {
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
      int srcY = (int) std::floor(dstY/yScale);
      for (int c = 0 ; c < channels ; c++) {
	float v = 0.0;
	float totalW = 0.0;
	for (int y = -ySrcWindow ; y < ySrcWindow ; y++) {
	  int srcY2 = clamp(srcY + y, 0, srcHeight - 1);
	  float *srcRGB = tensor.dataAsArray({srcY2, srcX, 0});
	  float d = std::abs(static_cast<float>(y) / static_cast<float> (ySrcWindow));
	  float w = 1.0 - d; // Bi-Linear
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
  assert(shape.size() == 3); // 3D shape: HWC
  assert((channels == 1) || (channels == 3)); // grayscale or RGB
  assert((filter == RESIZE_FILTER_NEAREST_NEIGHBOR) || (channels == RESIZE_FILTER_BI_LINEAR));
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


} // namespace image
} // namespace blueoil

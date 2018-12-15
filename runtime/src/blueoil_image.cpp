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

/*
 * Resize Image
 */
Tensor ResizeHorizontal(Tensor &tensor, const int width,
			const enum ResizeFilter filter) {
    auto shape = tensor.shape();
    const int srcHeight = shape[0];
    const int srcWidth  = shape[1];
    const int channels  = shape[2];
    const int height = srcHeight;
    Tensor dstTensor({height, width, channels});
    float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
    int xSrcWindow;
    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
	xSrcWindow = 1;  // Nearest
    } else {
	xSrcWindow = std::floor(1/xScale);  // Bi-Liner
	xSrcWindow = (xSrcWindow < 2)? 2 :xSrcWindow;
    }
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
		    float w;
		    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
			w = (d<0.5)?1.0:0.0; // NearestNeighbor
		    } else {
			w = 1.0 - d; // Bi-Linear
		    }
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

Tensor ResizeVertical(Tensor &tensor, const int height,
		      const enum ResizeFilter filter) {
    auto shape = tensor.shape();
    const int srcHeight = shape[0];
    const int srcWidth  = shape[1];
    const int channels  = shape[2];
    const int width = srcWidth;
    Tensor dstTensor({height, width, channels});
    float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
    int ySrcWindow;
    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
	ySrcWindow = 1;  // Nearest
    } else {
	ySrcWindow = std::floor(1/yScale);  // Bi-Linear
	ySrcWindow = (ySrcWindow < 2)? 2 :ySrcWindow;
    }
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
		    float w;
		    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
			w = (d<0.5)?1.0:0.0; // NearestNeighbor
		    } else {
			w = 1.0 - d; // Bi-Linear
		    }
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
	dstImage = ResizeHorizontal(dstImage, width, filter);
    }
    if  (srcHeight != height) {
	dstImage = ResizeVertical(dstImage, height, filter);
    }
    return dstImage;
}   


} // namespace image
} // namespace blueoil

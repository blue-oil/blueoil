/* Copyright 2018 Leapmind Inc. */

#include <iostream>
#include <cmath>

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
 * return RGB float array
 */
float *Tensor_at(Tensor &tensor, const int x, const int y) {
    const int height = tensor.m_shape[0];
    const int width  = tensor.m_shape[1];
    const int channels  = tensor.m_shape[2];
    if ((channels != 1) && (channels != 3)) {
	throw std::invalid_argument("wrong channles != 1,3");
    }
    const int clamped_x = clamp(x, 0, width-1);
    const int clamped_y = clamp(y, 0, height-1);
    const int scanlineSize = width * channels;
    float *imagePtr = &(tensor.m_data[0]);
    float *scanlinePtr = imagePtr + clamped_y * scanlineSize;
    float *pixelPtr = scanlinePtr + clamped_x * channels;
    return pixelPtr;
}

Tensor Tensor_CHW_to_HWC(Tensor &tensor) {
    const int channels  = tensor.m_shape[0];
    const int height = tensor.m_shape[1];
    const int width  = tensor.m_shape[2];
    if ((channels != 1) && (channels != 3)) {
	throw std::invalid_argument("wrong channles != 1,3");
    }
    Tensor dstTensor({height, width, channels});
    int srcPlaneSize = width * height;
    float *srcImagePtr = &(tensor.m_data[0]);
    float *dstImagePtr = &(dstTensor.m_data[0]);
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
    int height = tensor.m_shape[0];
    int width  = tensor.m_shape[1];
    int channels  = tensor.m_shape[2];
    if ((channels != 1) && (channels != 3)) {
	throw std::invalid_argument("wrong channles != 1,3");
    }
    Tensor dstTensor({channels, height, width});
    float *srcImagePtr = &(tensor.m_data[0]);
    float *dstImagePtr = &(dstTensor.m_data[0]);
    for (int c = 0 ; c < channels ; c++) {
	float *srcPixelPtr = srcImagePtr + c;
	for (int y = 0 ; y < height ; y++) {
	    for (int x = 0 ; x < width ; x++) {
		*dstImagePtr = *srcPixelPtr;
		srcPixelPtr += 3;
		dstImagePtr++;
	    }
	}
    }
    return dstTensor;
}

/*
 * Resize Image
 */
Tensor ResizeHorizontal(Tensor &tensor, const int width,
			const enum ResizeFilter filter) {
    const int srcHeight = tensor.m_shape[0];
    const int srcWidth  = tensor.m_shape[1];
    const int channels  = tensor.m_shape[2];
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
		    float *srcRGB = blueoil::image::Tensor_at(tensor,
							      srcX + x, srcY);
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
		float *dstRGB = blueoil::image::Tensor_at(dstTensor,
							  dstX, dstY);
		dstRGB[c] = v / totalW;
	    }
	}
    }
    return dstTensor;
}

Tensor ResizeVertical(Tensor &tensor, const int height,
		      const enum ResizeFilter filter) {
    const int srcHeight = tensor.m_shape[0];
    const int srcWidth  = tensor.m_shape[1];
    const int channels  = tensor.m_shape[2];
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
		    float *srcRGB = blueoil::image::Tensor_at(tensor,
							      srcX, srcY + y);
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
		float *dstRGB = blueoil::image::Tensor_at(dstTensor,
							  dstX, dstY);
		dstRGB[c] = v / totalW;
	    }
	}
    }
    return dstTensor;
} 

Tensor Resize(const Tensor& image, const std::pair<int, int>& size,
	      const enum ResizeFilter filter) {
    int channels  = image.m_shape[2];
    const int width = size.first;
    const int height = size.second;
    if ((channels != 1) && (channels != 3)) { // neither grayscale nor RGB
	throw std::invalid_argument("wrong channles != 1,3");
    }
    if ((filter != RESIZE_FILTER_NEAREST_NEIGHBOR) &&
	(filter != RESIZE_FILTER_BI_LINEAR)) {
	throw std::invalid_argument("unknown ResizeFilter");
    }
    const int srcHeight = image.m_shape[0];
    const int srcWidth  = image.m_shape[1];
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

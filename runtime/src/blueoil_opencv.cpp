/* Copyright 2018 Leapmind Inc. */

#include <cassert>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_opencv.hpp"

namespace blueoil {
namespace opencv {

/*
 * accept BGR OpenCV Mat images (not RGB)
 */
Tensor Tensor_fromCVMat(cv::Mat img) {
    int width = img.cols;
    int height = img.rows;
    int channels = img.elemSize();
    assert((channels == 1) || (channels == 3)); // grayscale or RGB
    blueoil::Tensor tensor({height, width, channels});
    for (int y = 0 ; y < height ; y++) {
	for (int x = 0 ; x < width ; x++) {
	    float *tensorPixel = blueoil::image::Tensor_at(tensor, x, y);
	    uchar *imgPixel = &(img.data[ y * img.step + x * img.elemSize()]);
	    if (channels == 1) {
		tensorPixel[0] = imgPixel[0]; // I (grayscale)
	    } else {  // (channels == 3)
		tensorPixel[0] = imgPixel[2]; // R
		tensorPixel[1] = imgPixel[1]; // G
		tensorPixel[2] = imgPixel[0]; // B
	    }
	}
    }
    //
    return tensor;
}

/*
 * generate BGR OpenCV Mat images (not RGB)
 */
cv::Mat Tensor_toCVMat(Tensor &tensor) {
    auto shape = tensor.shape();
    int height = shape[0];
    int width  = shape[1];
    int channels = shape[2];
    cv::Mat img;
    assert((channels == 1) || (channels == 3)); // grayscale or RGB
    if (channels == 1) {
	img = cv::Mat::zeros(height, width, CV_8U);   // uchar[1] grayscale
    } else { //  (channels == 3)
	img = cv::Mat::zeros(height, width, CV_8UC3); // uchar[3] rgb color
    }
    for (int y = 0 ; y < height ; y++) {
	for (int x = 0 ; x < width ; x++) {
	    float *tensorPixel = blueoil::image::Tensor_at(tensor, x, y);
	    uchar *imgPixel = &(img.data[ y * img.step + x * img.elemSize()]);
	    if (channels == 1) {
		imgPixel[0] = tensorPixel[0]; // I (grayscale)
	    } else {  // (channels == 3)
		imgPixel[2] = tensorPixel[0]; // R
		imgPixel[1] = tensorPixel[1]; // G
		imgPixel[0] = tensorPixel[2]; // B
	    }
	}
    }
    return img;
}

}  // namespace opencv
}  // namespace blueoil



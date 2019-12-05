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
  assert((channels == 1) || (channels == 3));  // grayscale or RGB
  blueoil::Tensor tensor({height, width, channels});
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float *tensorPixel = tensor.dataAsArray({y, x, 0});
      uchar *imgPixel = &(img.data[ y * img.step + x * img.elemSize()]);
      if (channels == 1) {
        tensorPixel[0] = imgPixel[0];  // I (grayscale)
      } else {  // (channels == 3)
        tensorPixel[0] = imgPixel[2];  // R
        tensorPixel[1] = imgPixel[1];  // G
        tensorPixel[2] = imgPixel[0];  // B
      }
    }
  }
  //
  return tensor;
}

/*
 * generate BGR OpenCV Mat images (not RGB)
 */
cv::Mat Tensor_toCVMat(const Tensor &tensor) {
  auto shape = tensor.shape();
  int height = shape[0];
  int width  = shape[1];
  int channels = shape[2];
  cv::Mat img;
  assert((channels == 1) || (channels == 3));  // grayscale or RGB
  if (channels == 1) {
    img = cv::Mat::zeros(height, width, CV_8U);    // uchar[1] grayscale
  } else {  //  (channels == 3)
    img = cv::Mat::zeros(height, width, CV_8UC3);  // uchar[3] rgb color
  }
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const float *tensorPixel = tensor.dataAsArray({y, x, 0});
      uchar *imgPixel = &(img.data[ y * img.step + x * img.elemSize()]);
      if (channels == 1) {
        imgPixel[0] = tensorPixel[0];  // I (grayscale)
      } else {  // (channels == 3)
        imgPixel[2] = tensorPixel[0];  // R
        imgPixel[1] = tensorPixel[1];  // G
        imgPixel[0] = tensorPixel[2];  // B
      }
    }
  }
  return img;
}


}  // namespace opencv
}  // namespace blueoil

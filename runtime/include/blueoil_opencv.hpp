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

#ifndef RUNTIME_INCLUDE_BLUEOIL_OPENCV_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_OPENCV_HPP_

#include <opencv2/opencv.hpp>

#include "blueoil.hpp"

namespace blueoil {
namespace opencv {

Tensor Tensor_fromCVMat(cv::Mat img);
cv::Mat Tensor_toCVMat(const Tensor &tensor);

}  // namespace opencv
}  // namespace blueoil

#endif  // RUNTIME_INCLUDE_BLUEOIL_OPENCV_HPP_

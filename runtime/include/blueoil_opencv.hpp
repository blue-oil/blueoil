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

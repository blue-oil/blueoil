/* Copyright 2018 Leapmind Inc. */
#ifndef RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_

#include "blueoil.hpp"

namespace blueoil {
namespace image {

float *Tensor_at(Tensor &tensor, const int x, const int y); // return RGB
Tensor Tensor_CHW_to_HWC(Tensor &tensor);
Tensor Tensor_HWC_to_CHW(Tensor &tensor);

enum ResizeFilter {
		   RESIZE_FILTER_NEAREST_NEIGHBOR = 1,
		   RESIZE_FILTER_BI_LINEAR = 2,
};

Tensor Resize(const Tensor& image, const std::pair<int, int>& size,
	      const enum ResizeFilter filter);
    
} // namespace image
} // namespace blueoil

#endif  // RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_

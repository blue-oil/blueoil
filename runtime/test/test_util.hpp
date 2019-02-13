#ifndef RUNTIME_INCLUDE_TENSOR_UTIL_HPP_
#define RUNTIME_INCLUDE_TENSOR_UTIL_HPP_

#include "blueoil.hpp"

namespace blueoil {
namespace util {

Tensor Tensor_CHW_to_HWC(Tensor &tensor);
Tensor Tensor_HWC_to_CHW(Tensor &tensor);

} // namespace util
} // namespace blueoil

#endif // RUNTIME_INCLUDE_TENSOR_UTIL_HPP_

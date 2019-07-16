#ifndef RUNTIME_TEST_TEST_UTIL_HPP_
#define RUNTIME_TEST_TEST_UTIL_HPP_

#include "blueoil.hpp"

namespace blueoil {
namespace util {

Tensor Tensor_CHW_to_HWC(const Tensor &tensor);
Tensor Tensor_HWC_to_CHW(const Tensor &tensor);

}  // namespace util
}  // namespace blueoil

#endif  // RUNTIME_TEST_TEST_UTIL_HPP_

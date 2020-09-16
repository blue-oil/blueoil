/* Copyright 2020 The Blueoil Authors. All Rights Reserved.  
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

#include "func/mul.h"
#include "gtest/gtest.h"

#include <array>
#include <range/irange.hpp>
#include <range/numeric.hpp>

constexpr std::size_t H = 3, W = 4, C = 5;
TEST(ConverterTest, FuncMul0){
    std::array<std::int64_t, H * W * C> a, b, c, d;
    helper::iota(a, 2);
    helper::iota(b, 2);
    for(auto i: helper::irange(a.size())){
        d[i] = a[i] * b[i];
    }
    using view_t = TensorView<std::int64_t, MemoryLayout::HWC>;
    view_t view0(a.data(), {H, W, C});
    view_t view1(b.data(), {H, W, C});
    view_t view2(c.data(), {H, W, C});
    view_t view3(d.data(), {H, W, C});
    func_Mul(view0, view1, view2);

    EXPECT_EQ(c, d);
}

TEST(ConverterTest, FuncMul1){
    std::array<std::int64_t, H * W * C> a, c, d;
    std::array<std::int64_t, W * C> b;
    helper::iota(a, 2);
    helper::iota(b, 2);
    TensorView<std::int64_t, MemoryLayout::HWC> view0(a.data(), {H, W, C});
    TensorView<std::int64_t, MemoryLayout::WC>  view1(b.data(), {W, C});
    TensorView<std::int64_t, MemoryLayout::HWC> view2(c.data(), {H, W, C});
    TensorView<std::int64_t, MemoryLayout::HWC> view3(d.data(), {H, W, C});
    for(auto i: helper::irange(H))
    for(auto j: helper::irange(W))
    for(auto k: helper::irange(C)){
        view3(i, j, k) = view0(i, j, k) * view1(j, k);
    }
    func_Mul(view0, view1, view2);

    EXPECT_EQ(c, d);
}
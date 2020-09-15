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

#include "../../../../../../blueoil/converter/templates/include/func/impl/binary_op.h"
#include "gtest/gtest.h"

#include <array>
#include <functional>
#include <range/numeric.hpp>
#include <range/irange.hpp>

constexpr std::size_t H = 3, W = 4, C = 5;
using func_t = std::multiplies<>;
constexpr auto func = func_t();

TEST(ConverterTest, BinaryOp0)
{
    std::array<std::int64_t, H * W * C> a, b, c, d;
    helper::iota(a, 1);
    helper::iota(b, 1);
    for(auto i: helper::irange(a.size())){
        d[i] = func(a[i], b[i]);
    }
    using view_t = TensorView<std::int64_t, MemoryLayout::HWC>;
    view_t view0(a.data(), {H, W, C});
    view_t view1(b.data(), {H, W, C});
    view_t view2(c.data(), {H, W, C});
        
    dlk::impl::binary_op<std::int64_t, MemoryLayout::HWC, MemoryLayout::HWC, func_t> bin_op;
    bin_op(view0, view1, view2, func);

    EXPECT_EQ(c, d);
}

TEST(ConverterTest, BinaryOp1)
{
    std::array<std::int64_t, H * W * C> a;
    std::array<std::int64_t, W * C> b;
    std::array<std::int64_t, H * W * C> c, d;
    helper::iota(a, 1);
    helper::iota(b, 1);
    TensorView<std::int64_t, MemoryLayout::HWC> view0(a.data(), {H, W, C});
    TensorView<std::int64_t, MemoryLayout::WC>  view1(b.data(), {W, C});
    TensorView<std::int64_t, MemoryLayout::HWC> view2(c.data(), {H, W, C});
    TensorView<std::int64_t, MemoryLayout::HWC> view3(d.data(), {H, W, C});

    for(auto i: helper::irange(H))
    for(auto j: helper::irange(W))
    for(auto k: helper::irange(C)){
        view3(i, j, k) = func(view0(i, j, k), view1(j, k));
    }

    dlk::impl::binary_op<std::int64_t, MemoryLayout::HWC, MemoryLayout::WC, func_t> bin_op;
    bin_op(view0, view1, view2, func);

    EXPECT_EQ(c, d);
}

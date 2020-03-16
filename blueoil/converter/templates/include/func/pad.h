#ifndef DLK_FUNC_PAD_H_INCLUDED
#define DLK_FUNC_PAD_H_INCLUDED

#include "types.h"
#include "tensor_view.h"

void func_Pad(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<int32_t, MemoryLayout::Padding>& padding,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output);

#endif // DLK_FUNC_PAD_H_INCLUDED

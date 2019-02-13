#include "tensor_util.hpp"
#include "blueoil.hpp"


namespace blueoil {
namespace util {

Tensor Tensor_CHW_to_HWC(Tensor &tensor) {
    auto shape = tensor.shape();
    const int channels  = shape[0];
    const int height = shape[1];
    const int width  = shape[2];
    Tensor dstTensor({height, width, channels});
    int srcPlaneSize = width * height;
    float *srcImagePtr = tensor.dataAsArray();
    float *dstImagePtr = dstTensor.dataAsArray();
    for (int y = 0 ; y < height ; y++) {
	for (int x = 0 ; x < width ; x++) {
	    float *srcPixelPtr0 = srcImagePtr + x + (y * height);
	    for (int c = 0 ; c < channels ; c++) {
		float *srcPixelPtr = srcPixelPtr0 + (c * srcPlaneSize);
		*dstImagePtr = *srcPixelPtr;
		dstImagePtr++;
	    }
	}
    }
    return dstTensor;
}

Tensor Tensor_HWC_to_CHW(Tensor &tensor) {
    auto shape = tensor.shape();
    int height = shape[0];
    int width  = shape[1];
    int channels = shape[2];
    Tensor dstTensor({channels, height, width});
    float *srcImagePtr = tensor.dataAsArray();
    float *dstImagePtr = dstTensor.dataAsArray();
    for (int c = 0 ; c < channels ; c++) {
	float *srcPixelPtr = srcImagePtr + c;
	for (int y = 0 ; y < height ; y++) {
	    for (int x = 0 ; x < width ; x++) {
		*dstImagePtr = *srcPixelPtr;
		srcPixelPtr += channels;
		dstImagePtr++;
	    }
	}
    }
    return dstTensor;
}

} // namespace util
} // namespace blueoil

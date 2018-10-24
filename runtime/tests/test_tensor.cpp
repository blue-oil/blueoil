#include <iostream>
#include "blueoil.hpp"
#include "blueoil_data_processor.hpp"

int main(void) {
    float mat[][3] = {
		      {1, 2, 3},
		      {7, 8, 9}
    };
    blueoil::Tensor tensor = blueoil::Tensor::zeros({2, 3});
    tensor.dump();
    blueoil::Tensor tensor2 = blueoil::Tensor::array({2, 3}, (float *) mat);
    tensor2.dump();
    tensor.data[0] = 1;
    tensor.data[1] = 2;
    tensor.data[2] = 3;
    tensor.data[3] = 7;
    tensor.data[4] = 8;
    tensor.data[5] = 9;
    tensor.dump();
    std::cout << tensor.allequal(tensor2) << std::endl;
    std::cout << tensor.allclose(tensor2) << std::endl;
    tensor.data[3] = 0; // NG
    std::cout << tensor.allequal(tensor2) << std::endl;
    std::cout << tensor.allclose(tensor2) << std::endl;
    return 0;
}


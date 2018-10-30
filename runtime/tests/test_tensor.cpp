#include <cstdlib>
#include <iostream>
#include "blueoil.hpp"
#include "blueoil_data_processor.hpp"

int test_tensor() {
    blueoil::Tensor tensor= blueoil::Tensor::zeros({2, 3});
    tensor.data[0] = 1;
    tensor.data[1] = 2;
    tensor.data[2] = 3;
    tensor.data[3] = 7;
    tensor.data[4] = 8;
    tensor.data[5] = 9;
    float expected_data[][3] = {
				{1, 2, 3},
				{7, 8, 9}
    };
    blueoil::Tensor expect = blueoil::Tensor::array({2, 3},
						    (float *) expected_data);
    if ((tensor.allequal(expect) == false) ||
	(tensor.allclose(expect) == false)) {
	std::cerr << "tensor_test: output != expect" << std::endl;
	tensor.dump();
	expect.dump();
	return EXIT_FAILURE;
    }
    tensor.data[3] = 11; // NG data
    if ((tensor.allequal(expect) == true) ||
	(tensor.allclose(expect) == true)) {
	std::cerr << "tensor_test: output(modified) == expect" << std::endl;
	tensor.dump();
	expect.dump();
	return EXIT_FAILURE;
     }
    return EXIT_SUCCESS;
}


int main(void) {
    int status_code = test_tensor();
    std::exit(status_code);
}


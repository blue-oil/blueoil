#include <cstdlib>
#include <iostream>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_opencv.hpp"

float test_input[3][8][8] =
    { { // Red
       {255, 0, 0, 0, 0, 0, 0, 0},
       {0, 255, 0, 0, 0, 0, 0, 0},
       {0, 0, 100, 0, 0, 0, 0, 0},
       {0, 0, 0, 100, 0, 0, 0, 0},
       {0, 0, 0, 0, 100, 0, 0, 0},
       {0, 0, 0, 0, 0, 100, 0, 0},
       {0, 0, 0, 0, 0, 0, 255, 0},
       {0, 0, 0, 0, 0, 0, 0, 255}
       },
      { // Green
       {0, 0, 0, 0, 0, 0, 0, 255},
       {0, 0, 0, 0, 0, 0, 255, 0},
       {0, 0, 0, 0, 0, 100, 0, 0},
       {0, 0, 0, 0, 100, 0, 0, 0},
       {0, 0, 0, 100, 0, 0, 0, 0},
       {0, 0, 100, 0, 0, 0, 0, 0},
       {0, 255, 0, 0, 0, 0, 0, 0},
       {255, 0, 0, 0, 0, 0, 0, 0}
      },
      { // Blue
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0},
       {255,255,255,100,100,255,255,255},
       {255,255,255,100,100,255,255,255},
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0},
       {  0,  0,  0,255,255,  0,  0,  0}
      } };

float test_expect[3][4][4] =
    { { // Red
       {255,  0, 0,   0},
       {  0,100, 0,   0},
       {  0,  0,100,  0},
       {  0,  0,  0,255}
       },
      { // Green
       {  0,  0,  0,  0},
       {  0,  0,  0,  0},
       {  0,  0,  0,  0},
       {  0,  0,  0,  0}
      },
      { // Blue
       {  0,  0,255,  0},
       {  0,  0,255,  0},
       {255,255,100,255},
       {  0,  0,255,  0}
      } };

int test_resize() {
    // CHW (3-channel, height, width)
    int width = 4, height = 4;
    blueoil::Tensor input({3, 8, 8}, (float *)test_input);
    blueoil::Tensor expect({3, 4, 4}, (float *)test_expect);
    input = blueoil::image::Tensor_CHW_to_HWC(input);
    expect = blueoil::image::Tensor_CHW_to_HWC(expect);
    blueoil::Tensor output = blueoil::image::Resize(input, width, height,
						    blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR);
    if (! output.allclose(expect)) {
	std::cerr << "test_resize: output != expect" << std::endl;
	blueoil::image::Tensor_HWC_to_CHW(output).dump();
	blueoil::image::Tensor_HWC_to_CHW(expect).dump();
	return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int command_resize(int argc, char **argv) {
    char *infile = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    blueoil::image::ResizeFilter filter = blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR;
    if ((width <= 0) || (height <= 0)) {
	std::cerr << "width <= 0 || height <= 0" << std::endl;
	return EXIT_FAILURE;
    }
    if (5 < argc) {
	int f = atoi(argv[4]);
	if ((f != blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR) &&
	    ( f != blueoil::image::RESIZE_FILTER_BI_LINEAR)) {
	    std::cerr << "unknown filter:" << f << std::endl;
	    return EXIT_FAILURE;
	}
	filter = static_cast<blueoil::image::ResizeFilter>(f);
    }
    char *outfile = argv[4];
    std::cout << "infile:" << infile << " width:" << width <<
	" height:" << height << " outfile:" << outfile << std::endl;
    cv::Mat img = cv::imread(infile, 1); // 1:force to RGB format
    if (img.data == NULL) {
	std::cerr << "can't open image file:" << infile <<std::endl;
	return EXIT_FAILURE;
    }
    blueoil::Tensor input = blueoil::opencv::Tensor_fromCVMat(img);
    blueoil::Tensor output = blueoil::image::Resize(input, width, height,
						    filter);
    cv::Mat img2 = blueoil::opencv::Tensor_toCVMat(output);
    cv::imwrite(outfile, img2);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    int status_code = 0;
    if ((argc == 5) || (argc == 6)) {
	status_code = command_resize(argc, argv);
	std::exit(status_code);	
    }
    if (argc == 1) {
	status_code = test_resize();
	if (status_code != EXIT_FAILURE) {
	    std::exit(status_code);
	}
	std::exit(EXIT_SUCCESS);
    }
    std::cerr <<
	"Usage: " << argv[0] << " # unit test. no news is good news" << std::endl <<
	"Usage: " << argv[0] << " <input image> <dstWidth> <dstHeight> <output  image> [<filter(1:nn,2:bl)>]" << std::endl;
    std::exit(EXIT_SUCCESS);
}

/* Copyright 2018 Leapmind Inc. */
#ifndef RUNTIME_INCLUDE_DCORE_HPP_
#define RUNTIME_INCLUDE_DCORE_HPP_


#include <string>
#include <vector>
#include <functional>


// TODO(wakisaka): Should use netowrk.h from dlk. But dlk's netwrok.h has so many dependancies.
extern "C" {
  class Network;
  Network* network_create();
  void network_delete(Network *nn);
  bool network_init(Network *nn);
  int network_get_input_rank(Network *nn);
  int network_get_output_rank(Network *nn);
  void network_get_input_shape(Network *nn, int *shape);
  void network_get_output_shape(Network *nn, int *shape);
  void network_run(Network *nn, float *input, float *output);
}


namespace blueoil {
struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;
};


// typedef Tensor (*TensorFunction)(Tensor&);
typedef std::function<Tensor(const Tensor& input)> Processor;

class Predictor {
 public:
  std::string task;
  std::vector<std::string> classes;
  std::vector<int> expected_input_shape;

  Tensor Run(const Tensor& image);

  // constructor
  explicit Predictor(const std::string& meta_yaml_path);


 private:
  // void SetupNetwork(const std::string dlk_so_lib_path);
  void SetupNetwork();
  void SetupMeta(const std::string& meta_yaml_path);
  Tensor RunPreProcess(const Tensor& input);
  Tensor RunPostProcess(const Tensor& input);

  Network* net_;
  // NetworkRun network_run;
  std::vector<int> network_input_shape_;
  std::vector<int> network_output_shape_;
  std::vector<int> image_size_;

  std::vector<Processor> pre_process_;
  std::vector<Processor> post_process_;
};

namespace box_util {

struct Box {
  float x;  // left
  float y;  // top
  float w;
  float h;
};

struct DetectedBox:Box {
  int class_id;
  float score;
};

// format output tensor to detected box to easy use for the user. be able to do on object detection task.
std::vector<DetectedBox> FormatDetectedBox(Tensor output_tensor);

}  // namespace box_util
}  // namespace blueoil

#endif  // RUNTIME_INCLUDE_DCORE_HPP_

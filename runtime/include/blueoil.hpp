#ifndef RUNTIME_INCLUDE_BLUEOIL_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_HPP_


#include <string>
#include <vector>
#include <functional>


// TODO(wakisaka): Should use netowrk.h from dlk. But dlk's netwrok.h has so many dependancies.
extern "C" {
  class Network;
  Network* network_create();
  void network_delete(Network *nn);
  bool network_init(Network *nn);
  int network_get_input_rank(const Network *nn);
  int network_get_output_rank(const Network *nn);
  void network_get_input_shape(const Network *nn, int *shape);
  void network_get_output_shape(const Network *nn, int *shape);
  void network_run(Network *nn, const float *input, float *output);
}


namespace blueoil {
class Tensor {
private:
  std::vector<int> shape_;
  std::vector<float> data_;
  int shapeVolume();
public:
  Tensor(std::vector<int> shape);
  Tensor(std::vector<int> shape, std::vector<float> data);
  Tensor(std::vector<int> shape, float *data);
  Tensor(const Tensor &tensor);
  std::vector<int> shape() const;
  std::vector<float> & data();
  const float *dataAsArray() const;
  const float *dataAsArray(std::vector<int> indices) const;
  float *dataAsArray();
  float *dataAsArray(std::vector<int> indices);
  void dump() const;
  std::vector<float>::const_iterator begin() const;
  std::vector<float>::const_iterator end() const;
  std::vector<float>::iterator begin();
  std::vector<float>::iterator end();
  bool allequal(const Tensor &tensor) const;
  bool allclose(const Tensor &tensor) const;
  // rtol: relative tolerance parameter
  // atol: absolute tolerance parameter
  bool allclose(const Tensor &tensor, float rtol, float atol) const;
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
std::vector<DetectedBox> FormatDetectedBox(const Tensor& output_tensor);

}  // namespace box_util
}  // namespace blueoil

#endif  // RUNTIME_INCLUDE_BLUEOIL_HPP_

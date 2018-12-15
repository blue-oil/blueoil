#include <dlfcn.h>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "blueoil.hpp"
#include "blueoil_data_processor.hpp"

#include "yaml-cpp/yaml.h"


namespace blueoil {


int Tensor::shapeVolume() {
    return this->shapeVolume(m_shape);
}

int Tensor::shapeVolume(std::vector<int> shape) {
    return std::accumulate(shape.begin(), shape.end(),
			   1, std::multiplies<int>());
}


Tensor::Tensor(std::vector<int> shape)
    : m_shape(shape),
      m_data(std::vector<float>(this->shapeVolume(std::move(shape)), 0)) {
}

Tensor::Tensor(std::vector<int> shape, std::vector<float> data)
    : m_shape(std::move(shape)),
      m_data(std::move(data)) {
}

Tensor::Tensor(std::vector<int> shape, float *arr)
    : m_shape(shape),
      m_data(std::vector<float>(arr,
				arr + this->shapeVolume(std::move(shape)))) {
}

Tensor::Tensor(const Tensor &tensor)
    : m_shape(tensor.m_shape),
      m_data(tensor.m_data) {
}


std::vector<int> Tensor::shape() const {
    return m_shape;
}

std::vector<float> &Tensor::data() {
  return m_data;
}

const float *Tensor::dataAsArray() const {
  if (this->m_shape.size() == 0) {
    throw std::invalid_argument("Tensor have no shape");
  }
  return &(m_data[0]);
}

const float *Tensor::dataAsArray(std::vector<int> indices) const {
    if (this->m_shape.size() != indices.size() ) {
        throw std::invalid_argument("shape.size != indices.size");
    }
    int i = 0;
    for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
	if ((*itr < 0) || (this->m_shape[i] <= *itr)) {
	    throw std::invalid_argument("indices out of shape range");
	}
    }
    int offset = 0, size = this->m_data.size();
    i = 0;
    for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
	size /= this->m_shape[i];
        offset += (*itr) * size;
    }
    return this->m_data.data() + offset;
}

float *Tensor::dataAsArray() {
  if (this->m_shape.size() == 0) {
    throw std::invalid_argument("Tensor have no shape");
  }
  return &(m_data[0]);
}

float *Tensor::dataAsArray(std::vector<int> indices) {
    if (this->m_shape.size() != indices.size() ) {
        throw std::invalid_argument("shape.size != indices.size");
    }
    int i = 0;
    for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
	if ((*itr < 0) || (this->m_shape[i] <= *itr)) {
	    throw std::invalid_argument("indices out of shape range");
	}
    }
    int offset = 0, size = this->m_data.size();
    i = 0;
    for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
	size /= this->m_shape[i];
        offset += (*itr) * size;
    }
    return this->m_data.data() + offset;
}


static void Tensor_shape_dump(const std::vector<int>& shape) {
    std::cout << "shape:";
    for (auto itr = shape.begin(); itr != shape.end(); ++itr) {
        std::cout << *itr << " ";
    }
    std::cout << std::endl;
}

static void Tensor_data_dump(const float *data, const std::vector<int>& shape){
    if (shape.size() == 1) { // 1-D array
        auto itr = shape.begin();
        int n = *itr;
        for (int i = 0 ; i < n ;  i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } else if (shape.size() == 2) { // 2-D arra
        auto itr = shape.begin();
        int w = *itr;
        int c = *(itr+1);
        for (int x = 0 ; x < w ; x++) {
            for (int i = 0 ; i < c ; i++) {
                std::cout << data[c*x + i] << " ";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    } else { // 3-D over to recursive
        auto itr = shape.begin();
        int n  = *itr;
        int stride = 1;
        for (itr++; itr != shape.end(); ++itr) {
            stride *= *itr;
        }
        std::vector<int> shape2 = shape;
        shape2.erase(shape2.begin());
        for (int i = 0 ; i < n ; i++) {
            Tensor_data_dump(data + i*stride, shape2);
        }
    }
}

// dump N-dimentional array
void Tensor::dump() const {
    Tensor_shape_dump(m_shape);
    Tensor_data_dump(&(m_data[0]), m_shape);
}


std::vector<float>::const_iterator Tensor::begin() const {
    return this->m_data.begin();
}

std::vector<float>::const_iterator Tensor::end() const {
    return this->m_data.end();
}

std::vector<float>::iterator Tensor::begin() {
    return this->m_data.begin();
}

std::vector<float>::iterator Tensor::end() {
    return this->m_data.end();
}


// all elements exact equals check.
bool Tensor::allequal(const Tensor &tensor) const {
    if ((m_shape != tensor.m_shape) || (m_data != tensor.m_data)) {
        return false;
    }
    return true;
}


// all elements nealy equals check.
bool Tensor::allclose(const Tensor &tensor) const {
    float rtol=1.e-5, atol=1.e-8; // same as numpy isclose
    return this->allclose(tensor, rtol, atol);
}

bool Tensor::allclose(const Tensor &tensor, float rtol, float atol) const {
    if (m_shape != tensor.m_shape) {
        return false;
    }
    int n = this->m_data.size();
    for (int i = 0 ; i < n ; i++) {
	float a = m_data[i];
	float b = tensor.m_data[i];
	if (std::abs(a - b) > (atol + rtol * std::abs(b))) {
	    return false;
	}
    }
    return true;
}


// mapping process node to functions vector.
void MappingProcess(const YAML::Node processors_node, std::vector<Processor>* functions) {
  switch (processors_node.Type()) {
    case YAML::NodeType::Null: {
      break;
    }

    case YAML::NodeType::Sequence: {
      for (const YAML::Node& process_node : processors_node) {
        for (const auto& key_val : process_node) {
          const auto& method_name = key_val.first.as<std::string>();
          const auto& method_params = key_val.second;

          // pre process.
          if (method_name == "DivideBy255") {
            Processor tmp = std::move(data_processor::DivideBy255);
            functions->push_back(std::move(tmp));

          } else if (method_name == "Resize" || method_name == "ResizeWithGtBoxes") {
            std::pair<int, int> size = method_params["size"].as<std::pair<int, int>>();
            Processor tmp = std::bind(data_processor::Resize, std::placeholders::_1, size);
            functions->push_back(std::move(tmp));

          // post process.
          } else if (method_name == "FormatYoloV2") {
            auto params = method_params.as<data_processor::FormatYoloV2Parameters>();
            Processor tmp = std::bind<Tensor(const Tensor&, const data_processor::FormatYoloV2Parameters&)>
                            (data_processor::FormatYoloV2, std::placeholders::_1, std::move(params));
            functions->push_back(std::move(tmp));

          } else if (method_name == "ExcludeLowScoreBox") {
            auto threshold = method_params["threshold"].as<float>();
            Processor tmp = std::bind(data_processor::ExcludeLowScoreBox, std::placeholders::_1,
                                      threshold);
            functions->push_back(std::move(tmp));

          } else if (method_name == "NMS") {
            auto params = method_params.as<data_processor::NMSParameters>();
            Processor tmp = std::bind<Tensor(const Tensor&, const data_processor::NMSParameters&)>
                            (data_processor::NMS, std::placeholders::_1, std::move(params));
            functions->push_back(std::move(tmp));

          } else {
            // TODO(wakisaka): error handle.
            std::cout <<  method_name << " is not register method: " << std::endl;
            exit(1);
          }
        }
      }
      break;
    }

    default: {
      // TODO(wakisaka): error handle.
      break;
    }
  }
}


void Predictor::SetupNetwork() {
  net_ = network_create();
  bool ret = network_init(net_);

  if (ret == false) {
    std::cout << "network init error" << std::endl;
    exit(1);
  }

  const int input_rank = network_get_input_rank(net_);
  const int output_rank = network_get_output_rank(net_);

  network_input_shape_.resize(input_rank);
  network_output_shape_.resize(output_rank);

  network_get_input_shape(net_, network_input_shape_.data());
  network_get_output_shape(net_, network_output_shape_.data());

  this->expected_input_shape = network_input_shape_;
}


Predictor::Predictor(const std::string& meta_yaml_path) {
  SetupNetwork();
  SetupMeta(meta_yaml_path);
  // TODO(wakisaka): check network input shape is the same as meta's image size.
  // TODO(wakisaka): check network output shape is the same as meta's number of class when type is classsification.
}

void Predictor::SetupMeta(const std::string& meta_yaml_path) {
  YAML::Node meta = YAML::LoadFile(meta_yaml_path.c_str());

  this->task = meta["TASK"].as<std::string>();

  std::vector<int> image_size_ = meta["IMAGE_SIZE"].as<std::vector<int>>();

  this->classes = meta["CLASSES"].as<std::vector<std::string>>();

  YAML::Node pre_processor_node = meta["PRE_PROCESSOR"];
  MappingProcess(pre_processor_node, &pre_process_);

  YAML::Node post_processor_node = meta["POST_PROCESSOR"];
  MappingProcess(post_processor_node, &post_process_);
}


Tensor Predictor::RunPreProcess(const Tensor& input) {
  Tensor tmp = input;
  for (Processor process : pre_process_) {
    tmp = process(tmp);
  }

  return tmp;
}

Tensor Predictor::RunPostProcess(const Tensor& input) {
  Tensor tmp = input;
  for (Processor process : post_process_) {
    tmp = process(tmp);
  }

  return tmp;
}

Tensor Predictor::Run(const Tensor& image) {
  Tensor pre_processed = RunPreProcess(image);

  // build network output tensor.
  Tensor n_output(network_output_shape_);

  network_run(net_, pre_processed.dataAsArray(), n_output.dataAsArray());

  Tensor post_processed = RunPostProcess(n_output);

  return post_processed;
}


namespace box_util {

// TODO(wakiska): implement this func.
std::vector<DetectedBox> FormatDetectedBox(const blueoil::Tensor& output_tensor);
}  // namespace box_util

}  // namespace blueoil


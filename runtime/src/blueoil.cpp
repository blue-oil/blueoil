/* Copyright 2019 Leapmind Inc. */

#include <dlfcn.h>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdio>

#include "blueoil.hpp"
#include "blueoil_data_processor.hpp"

#include "yaml-cpp/yaml.h"


namespace blueoil {
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
  const int size = std::accumulate(
      network_output_shape_.begin(), network_output_shape_.end(), 1, std::multiplies<int>());
  Tensor n_output = {
    std::vector<float>(size, 0),
    network_output_shape_
  };

  network_run(net_, pre_processed.data.data(), n_output.data.data());

  Tensor post_processed = RunPostProcess(n_output);

  return post_processed;
}


namespace box_util {

// TODO(wakiska): implement this func.
std::vector<DetectedBox> FormatDetectedBox(blueoil::Tensor output_tensor);
}  // namespace box_util

}  // namespace blueoil


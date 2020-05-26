/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

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

#ifndef RUNTIME_INCLUDE_BLUEOIL_DATA_PROCESSOR_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_DATA_PROCESSOR_HPP_


#include <string>
#include <vector>
#include <functional>
#include <utility>


#include "blueoil.hpp"

#include "yaml-cpp/yaml.h"

namespace blueoil {
namespace data_processor {
// pre process.
Tensor Resize(const Tensor& image, const std::pair<int, int>& size);

Tensor DivideBy255(const Tensor& image);

Tensor PerImageStandardization(const Tensor& image);

// post process.

Tensor FormatYoloV2(const Tensor& input,
                    const std::vector<std::pair<float, float>>& anchors,
                    const int& boxes_per_cell,
                    const std::string& data_format,
                    const std::pair<int, int>& image_size,
                    const int& num_classes);
struct FormatYoloV2Parameters {
  std::vector<std::pair<float, float>> anchors;
  int boxes_per_cell;
  std::string data_format;
  std::pair<int, int> image_size;
  int num_classes;
};
Tensor FormatYoloV2(const Tensor& input, const FormatYoloV2Parameters& params);


Tensor ExcludeLowScoreBox(const Tensor& input, const float& threshold);

struct NMSParameters {
  std::vector<std::string> classes;
  float iou_threshold;
  int max_output_size;
  bool per_class;
};
Tensor NMS(const Tensor& input,
           const std::vector<std::string>& classes,
           const float& iou_threshold,
           const int& max_output_size,
           const bool& per_class);
Tensor NMS(const Tensor& input,
           const NMSParameters& params);


}  // namespace data_processor
}  // namespace blueoil


// Converters to function's prameter from yaml node.
// See https://github.com/jbeder/yaml-cpp/wiki/Tutorial#converting-tofrom-native-data-types
namespace YAML {

template<>
struct convert<blueoil::data_processor::FormatYoloV2Parameters> {
  static Node encode(const blueoil::data_processor::FormatYoloV2Parameters& params);
  static bool decode(const Node& node,
                     blueoil::data_processor::FormatYoloV2Parameters& params) {  // NOLINT
    params.anchors = node["anchors"].as<std::vector<std::pair<float, float>>>();
    params.boxes_per_cell = node["boxes_per_cell"].as<int>();
    params.data_format = node["data_format"].as<std::string>();
    params.image_size = node["image_size"].as<std::pair<int, int>>();
    params.num_classes = node["num_classes"].as<int>();

    return true;
  }
};

template<>
struct convert<blueoil::data_processor::NMSParameters> {
  static Node encode(const blueoil::data_processor::NMSParameters& params);
  static bool decode(const Node& node,
                     blueoil::data_processor::NMSParameters& params) {  // NOLINT
    params.classes = node["classes"].as<std::vector<std::string>>();
    params.iou_threshold = node["iou_threshold"].as<float>();
    params.max_output_size = node["max_output_size"].as<int>();
    params.per_class = node["per_class"].as<bool>();

    return true;
  }
};
}  // namespace YAML


#endif  // RUNTIME_INCLUDE_BLUEOIL_DATA_PROCESSOR_HPP_

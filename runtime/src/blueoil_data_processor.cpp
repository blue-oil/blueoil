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

#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <utility>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#include "blueoil_data_processor.hpp"

namespace blueoil {
namespace data_processor {

static std::vector<float> softmax(const float* xs, int num) {
  std::vector<float> r(num);

  float max_val = 0.0;
  for (int i = 0; i < num; i++) {
    max_val = std::max(xs[i], max_val);
  }

  float exp_sum = 0.0;
  for (int i = 0; i < num; i++) {
    float val = exp(xs[i] - max_val);
    r[i] = val;
    exp_sum += val;
  }

  for (int i = 0; i < num; i++) {
    r[i] /= exp_sum;
  }

  return r;
}

static float sigmoid(float x) {
  // The exp function becomes infinite at x around 89(float) or 710(double),
  // so branch and calculate.
  if (x > 0) {
    return 1.0 / (1.0 + exp(-x));
  } else {
    auto e = exp(x);
    return e / (1.0 + e);
  }
}

static float CalcIoU(const box_util::Box& a, const box_util::Box& b) {
  float left = std::max(a.x, b.x);
  float top = std::min(a.y+a.h, b.y + b.h);

  float right = std::min(a.x + a.w, b.x + b.w);
  float bottom = std::max(a.y, b.y);

  float inner_area = std::max(right - left, 0.f) * std::max(top - bottom, 0.f);
  float a_area = a.w * a.h;
  float b_area = b.w * b.h;
  float epsilon = 1e-10;

  float r = inner_area / (a_area + b_area - inner_area + epsilon);

  if (std::isnan(r)) {
    return 0.0;
  }
  if (r > 1.0) {
    return 1.0;
  } else if (r < 0.0) {
    return 0.0;
  }
  return r;
}

static box_util::Box ConvertBboxCoordinate(float x, float y, float w, float h, float k,
                                           const std::pair<float, float>& anchor,
                                           int nth_y, int nth_x,
                                           int num_cell_y, int num_cell_x) {
  box_util::Box r;
  float anchor_w = anchor.first;
  float anchor_h = anchor.second;

  float cy = (y + static_cast<float>(nth_y)) / num_cell_y;
  float cx = (x + static_cast<float>(nth_x)) / num_cell_x;

  r.h = exp(h) * static_cast<float>(anchor_h) / num_cell_y;
  r.w = exp(w) * static_cast<float>(anchor_w) / num_cell_x;

  r.y = cy - (r.h / 2);
  r.x = cx - (r.w / 2);

  return r;
}

Tensor Resize(const Tensor& image, const std::pair<int, int>& size) {
  const int width = size.first;
  const int height = size.second;
  return blueoil::image::Resize(image, width, height,
                                blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR);
}

Tensor DivideBy255(const Tensor& image) {
  Tensor out(image);

  auto div255 = [](float i) { return i/255; };
  std::transform(image.begin(), image.end(), out.begin(), div255);

  return out;
}

Tensor PerImageStandardization(const Tensor& image) {
  Tensor out(image);

  double sum = 0.0;
  double sum2 = 0.0;

  for (auto it : image) {
    sum += it;
    sum2 += it * it;
  }

  float mean = sum / image.size();
  float var = sum2 / image.size() - mean * mean;
  double sd = std::sqrt(var);
  float adjusted_sd = std::max(sd, 1.0 / std::sqrt(image.size()));
  auto standardization = [mean, adjusted_sd](float i) { return (i - mean) / adjusted_sd; };
  std::transform(image.begin(), image.end(), out.begin(), standardization);
  return out;
}


// convert yolov2's detection result to more easy format
// output coordinates are not translated into original image coordinates,
// since we can't know original image size.
Tensor FormatYoloV2(const Tensor& input,
                    const std::vector<std::pair<float, float>>& anchors,
                    const int& boxes_per_cell,
                    const std::string& data_format,
                    const std::pair<int, int>& image_size,
                    const int& num_classes) {
  // input shape must be NHWC, N == 1

  auto shape = input.shape();
  int num_cell_y = shape[1];
  int num_cell_x = shape[2];

  assert(shape[0] == 1);
  assert(shape.size() == 4);
  assert(input.size() % (num_cell_y * num_cell_x * anchors.size()) == 0);
  assert(static_cast<int>(anchors.size()) == boxes_per_cell);

  std::vector<int> output_shape = {1, num_cell_y * num_cell_x * boxes_per_cell * num_classes, 6};
  Tensor result(output_shape);

  int r_i = 0, r_delta = num_cell_y * num_cell_y * anchors.size();
  for (int i = 0; i < num_cell_y; i++) {
    for (int j = 0; j < num_cell_x; j++) {
      const float* predictions = input.dataAsArray({0, i, j, 0});
      for (size_t k = 0; k < anchors.size(); k++) {
        // is it ok to use softmax when num_classes == 1?
        std::vector<float> probs = softmax(predictions, num_classes);
        float conf = sigmoid(predictions[num_classes]);
        float x = sigmoid(predictions[num_classes+1]);
        float y = sigmoid(predictions[num_classes+2]);
        float w = predictions[num_classes+3];
        float h = predictions[num_classes+4];

        box_util::Box bbox_im = ConvertBboxCoordinate(x, y, w, h, k, anchors[k], i, j, num_cell_y, num_cell_x);

        int r_i2 = r_i;
        for (int c_i = 0; c_i < num_classes; c_i++) {
          float prob = probs[c_i];
          float score = prob * conf;
          auto p = result.dataAsArray({0, r_i2, 0});
          p[0] = bbox_im.x * image_size.first;
          p[1] = bbox_im.y * image_size.second;
          p[2] = bbox_im.w * image_size.first;
          p[3] = bbox_im.h * image_size.second;
          p[4] = c_i;
          p[5] = score;
          r_i2 += r_delta;
        }
        r_i++;
        predictions = predictions + (num_classes + 5);
      }
    }
  }

  return result;
}

Tensor FormatYoloV2(const Tensor& input, const FormatYoloV2Parameters& params) {
  return FormatYoloV2(input,
                      params.anchors,
                      params.boxes_per_cell,
                      params.data_format,
                      params.image_size,
                      params.num_classes);
}

Tensor ExcludeLowScoreBox(const Tensor& input, const float& threshold) {
  Tensor result(input);

  auto shape = input.shape();
  int num_predictions = shape[1];
  int compacted_num_predictions = 0;

  for (int i = 0; i < num_predictions; i++) {
    float* predictions = result.dataAsArray({0, i, 0});
    float score = predictions[5];
    if (score < threshold) {
      // delete entry
    } else {
      // remain entry
      float* compacted_predictions = result.dataAsArray({0, compacted_num_predictions, 0});
      std::memcpy(compacted_predictions, predictions, 6 * sizeof(float));
      compacted_num_predictions++;
    }
  }
  result.erase({0, compacted_num_predictions, 0}, {1, 0, 0});
  return result;
}

Tensor NMS(const Tensor& input,
           const std::vector<std::string>& classes,
           const float& iou_threshold,
           const int& max_output_size,
           const bool& per_class) {
  auto shape = input.shape();
  int num_predictions = shape[1];
  int num_classes = classes.size();

  std::vector<int> ids;
  for (int i = 0; i < num_predictions; i++) {
    ids.push_back(i);
  }

  // sort index by class_id & score.
  std::sort(ids.begin(), ids.end(),
            [input](const int& a, const int& b) -> bool {
              const float* prediction_a = input.dataAsArray({0, a, 0});
              float class_id_a = prediction_a[4];
              float score_a = prediction_a[5];
              const float* prediction_b = input.dataAsArray({0, b, 0});
              float class_id_b = prediction_b[4];
              float score_b = prediction_b[5];
              if (class_id_a != class_id_b) {
                return class_id_a < class_id_b;
              }
              return score_a > score_b;
            });

  Tensor result(input.shape());

  // sort vector elements by index
  for (int i = 0; i < num_predictions; i++) {
    const float* prediction = input.dataAsArray({0, ids[i], 0});
    float* store_location = result.dataAsArray({0, i, 0});
    auto class_id = prediction[4];
    if (class_id < num_classes) {
      std::memcpy(store_location, prediction, 6*sizeof(float));
    } else {
      break;
    }
  }

  // IoU overlap boxes marking for deletion
  if (per_class) {
    for (int class_id = 0; class_id < num_classes; class_id++) {
      int num_remain = 1;
      for (int i = 0; i < num_predictions; i++) {
        float* prediction_a = result.dataAsArray({0, i, 0});
        float score = prediction_a[5];
        if (score < 0.0) {
          break;
        } else if (score == 0.0) {
          continue;
        } else if (prediction_a[4] != class_id) {
          continue;
        }
        box_util::Box box_a = box_util::Box(prediction_a[0], prediction_a[1], prediction_a[2], prediction_a[3]);

        for (int j = i+1; j < num_predictions; j++) {
          float* prediction_b = result.dataAsArray({0, j, 0});
          if (prediction_b[4] != class_id) {
            continue;
          }
          if (max_output_size <= num_remain) {
            prediction_b[5] = 0.0;  // marked for deletion
            continue;
          }
          box_util::Box box_b = box_util::Box(prediction_b[0], prediction_b[1], prediction_b[2], prediction_b[3]);
          float iou = CalcIoU(box_a, box_b);

          if (iou < iou_threshold) {
            num_remain++;
          } else {
            prediction_b[5] = 0.0;  // marked for deletion
          }
        }
      }
    }
  } else {
    int num_remain = 1;
    for (int i = 0; i < num_predictions; i++) {
      float* prediction_a = result.dataAsArray({0, i, 0});
      float score = prediction_a[5];
      if (score < 0.0) {
        break;
      } else if (score == 0.0) {
        continue;
      }
      box_util::Box box_a = box_util::Box(prediction_a[0], prediction_a[1], prediction_a[2], prediction_a[3]);

      for (int j = i+1; j < num_predictions; j++) {
        float* prediction_b = result.dataAsArray({0, j, 0});
         if (max_output_size <= num_remain) {
           prediction_b[5] = 0.0;  // marked for deletion
           continue;
         }
        box_util::Box box_b = box_util::Box(prediction_b[0], prediction_b[1], prediction_b[2], prediction_b[3]);
        float iou = CalcIoU(box_a, box_b);

        if (iou < iou_threshold) {
          num_remain++;
        } else {
          prediction_b[5] = 0.0;  // marked for deletion
        }
      }
    }
  }

  // packing deletion area
  int j = 0;
  for (int i = 0; i < num_predictions; i++) {
    float* prediction = result.dataAsArray({0, i, 0});
    float score = prediction[5];
    if (score > 0.0) {
      float* store_location = result.dataAsArray({0, j, 0});
      std::memcpy(store_location, prediction, 6*sizeof(float));
      j++;
    }
  }
  // truncating deletion area
  result.erase({0, j, 0}, {1, 0, 0});
  return result;
}

Tensor NMS(const Tensor& input, const NMSParameters& params){
  return NMS(input,
             params.classes,
             params.iou_threshold,
             params.max_output_size,
             params.per_class);
};
}  // namespace data_processor
}  // namespace blueoil

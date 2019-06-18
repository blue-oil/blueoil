/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>

#include "time_measurement.h"

std::map<std::string, std::vector<Measurement::measure> > Measurement::times;
std::vector<std::string> Measurement::current_context;
std::vector<Measurement::Node*> Measurement::stack;
std::vector<std::unique_ptr<Measurement::Node>> Measurement::roots;


void Measurement::Start(const std::string &measure_name)
{
  #if defined FUNC_TIME_MEASUREMENT
  measure m;
  m.start = std::chrono::system_clock::now();
  m.end = std::chrono::system_clock::time_point();

  if (stack.empty()) {
      roots.emplace_back(std::make_unique<Node>(measure_name, 0));
      auto* current = roots.back().get();
      current->measurements.emplace_back(m);
      stack.emplace_back(current);
  }
  else {
      auto* parent = stack.back();
      auto it = parent->children.find(measure_name);
      if (it == parent->children.end()) {
        it = parent->children.emplace(std::piecewise_construct,
          std::forward_as_tuple(measure_name),
          std::forward_as_tuple(std::make_unique<Node>(measure_name, parent->children.size()))).first;
      }
      auto* current = it->second.get();
      current->measurements.emplace_back(m);
      stack.emplace_back(current);
  }

  times[measure_name].push_back(m);
  current_context.push_back(measure_name);
  #endif
}

void Measurement::Stop()
{
  #if defined FUNC_TIME_MEASUREMENT

  if(current_context.size() == 0 || times[current_context.back()].size() == 0) {
    std::cout << "ERROR: wrong Start/Stop pairs" << std::endl;
    return;
  }

  if (stack.empty()) {
    std::cout << "ERROR: wrong Start/Stop pairs" << std::endl;
    return;
  }

  stack.back()->measurements.back().end = std::chrono::system_clock::now();
  stack.pop_back();

  times[current_context.back()].back().end = std::chrono::system_clock::now();
  current_context.pop_back();
  #endif
}

void Measurement::Report()
{
  #if defined FUNC_TIME_MEASUREMENT
  for(auto it = times.begin(); it != times.end(); ++it)
  {
    std::cout << it->first << ",";
    double sum = 0.0;
    for(auto m = it->second.begin(); m != it->second.end(); ++m)
    {
      double t = std::chrono::duration_cast<TIME_ORDER>(m->end - m->start).count();
      std::cout << t << ",";
      sum +=t;
    }
    std::cout << "  sum:" << sum / 1000 << "ms";
    std::cout << std::endl;
  }

  std::cout << "---------------------------------------------------" << std::endl;
  for (auto& root: roots) {
    DumpTimeTree(*root, 0);
  }
  #endif
}

void Measurement::DumpTimeTree(const Node& node, int level) {
    std::cout << std::string(level * 2, '.') << node.name << " ";
    double sum = 0.0;
    for (auto& m : node.measurements) {
      double t = std::chrono::duration_cast<TIME_ORDER>(m.end - m.start).count();
      std::cout << t << ",";
      sum += t;
    }
    std::cout << "  sum:" << sum / 1000 << "ms";
    std::cout << std::endl;
    std::vector<const Node*> sorted;
    for (auto& p : node.children) {
      sorted.emplace_back(p.second.get());
    }
    std::sort(sorted.begin(), sorted.end(), [](auto* a, auto* b) {
      return a->position < b->position;
    });
    for (auto* child : sorted) {
      DumpTimeTree(*child, level + 1);
    }
}

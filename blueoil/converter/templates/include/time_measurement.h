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

#ifndef TIME_MEASURE_HEADER
#define TIME_MEASURE_HEADER

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_map>

#define TIME_ORDER std::chrono::microseconds


class Measurement
{
private:

  struct measure
  {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
  };

  struct Node {
    Node(const std::string& name, size_t position)
        : name(name), position(position) {
    }

    std::string name;
    size_t position;
    std::vector<measure> measurements;
    std::unordered_map<std::string, std::unique_ptr<Node>> children;
  };

  static std::map<std::string, std::vector<Measurement::measure> > times;
  static std::vector<std::string> current_context;

  static std::vector<Node*> stack;
  static std::vector<std::unique_ptr<Node>> roots;

  static void DumpTimeTree(const Node&, int level);

public:
  static void Start(const std::string &measure_name);
  static void Stop();
  static void Report();
};

#endif

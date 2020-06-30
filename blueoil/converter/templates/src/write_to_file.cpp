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

#include "types.h"
#include <string>
#include <fstream>

void write_to_file(const char *filename, int id, volatile int32_t* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " int32" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << data[i] << std::endl;
  outfile.close();
}

void write_to_file(const char *filename, int id, BIN_CONV_OUTPUT* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " BIN_CONV_OUTPUT" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << data[i] << std::endl;
  outfile.close();
}


void write_to_file(const char *filename, int id, QUANTIZED_NOT_PACKED* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " QUANTIZED_NOT_PACKED" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << (int) data[i] << std::endl;
  outfile.close();
}


void write_to_file(const char *filename, int id, float* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " float" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << data[i] << std::endl;
  outfile.close();
}

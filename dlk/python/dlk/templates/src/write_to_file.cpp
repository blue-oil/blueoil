#include "global.h"
#include <string>
#include <fstream>

void write_to_file(const char *filename, int id, volatile int32_t* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " int32" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << data[i] << std::endl;
  outfile.flush();
  outfile.close();
}

void write_to_file(const char *filename, int id, BIN_CONV_OUTPUT* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " BIN_CONV_OUTPUT" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << data[i] << std::endl;
  outfile.flush();
  outfile.close();
}


void write_to_file(const char *filename, int id, QUANTIZED_NOT_PACKED* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " QUANTIZED_NOT_PACKED" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << (int) data[i] << std::endl;
  outfile.flush();
  outfile.close();
}


void write_to_file(const char *filename, int id, float* data, int size) {
  std::string name(filename);
  name += std::to_string(id);

  std::ofstream outfile(name);
  outfile << __FUNCTION__ << " float" << std::endl;
  for(int i = 0; i < size; i++)
    outfile << i << "," << data[i] << std::endl;
  outfile.flush();
  outfile.close();
}

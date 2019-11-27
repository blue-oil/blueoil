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
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <numeric>
#include <functional>

#include "blueoil.hpp"
#include "blueoil_npy.hpp"


namespace blueoil {
namespace npy {

struct NPYheader_t {
  std::vector<int> shape;
  std::string datatype;  // |u1(uint8) or <f4(float32)
};

struct NPYheader_t readNPYheader(std::ifstream &fin);
template<typename T>
void readNPYdata(std::ifstream &fin, const struct NPYheader_t &nh,
                  T *imagedata);

Tensor Tensor_fromNPYFile(const std::string filename) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  std::stringstream ss;
  if (!fin) {
    ss << "Can't open file:" << filename;
    throw std::runtime_error(ss.str());
  }
  NPYheader_t nh = readNPYheader(fin);

  blueoil::Tensor tensor(nh.shape);
  readNPYdata(fin, nh, tensor.dataAsArray());
  return tensor;
}

// " ABC " => "ABC"
static std::string trim(const std::string str) {
  const char *trimChara = " \t\r\n";
  auto left = str.find_first_not_of(trimChara);
  if (left  == std::string::npos) {
    return "";
  }
  auto right = str.find_last_not_of(trimChara);
  return str.substr(left, right - left + 1);
}

// "{...}" => "..."
std::string extractInner(std::string strdata,
                         std::string leftSep, std::string rightSep) {
  auto braseFirstPos = strdata.find_first_of(leftSep);
  auto braseLastPos = strdata.find_last_of(rightSep);
  if ((braseFirstPos == std::string::npos) ||
      (braseLastPos == std::string::npos)) {
    return "";
  }
  return strdata.substr(braseFirstPos + 1, braseLastPos - braseFirstPos - 1);
}
std::string extractInner(std::string strdata, std::string sep) {
  return extractInner(strdata, sep, sep);
}

// "A,B,(C,D,E)" => "A","B","(C,E,E)"
std::vector<std::string> jsonCommaSplit(std::string strdata) {
  std::vector<std::string> jsonVector;
  std::string::size_type cur = 0, pos = 0;
  for (cur = 0 ; cur < strdata.size() && (pos != std::string::npos) ; cur++) {
    pos = strdata.find_first_of(",", cur);
    if (pos == std::string::npos) {
      pos = strdata.size();
    }
    auto bCur = strdata.find_first_of("(", cur);
    if (bCur != std::string::npos) {
      if (bCur < pos) {
        auto bCur2 = strdata.find_first_of(")", bCur + 1);
        if (bCur2 != std::string::npos) {
          auto bCur3 = strdata.find_first_of(",", bCur2 + 1);
          pos = bCur3;
        } else {
          throw std::runtime_error("can't closing bracket");
        }
      }
    }
    if (cur !=  pos) {
      auto value = strdata.substr(cur, pos - cur);
      jsonVector.push_back(trim(value));
    }
    cur = pos;
  }
  return jsonVector;
}
// "'A': B" =>  "A" => "B"
std::pair<std::string, std::string> jsonKeyValueSplit(std::string strdata) {
  auto pos = strdata.find_first_of(":", 0);
  if (pos == std::string::npos) {
    return std::pair<std::string, std::string>("", "");  // failed
  }
  std::string key = strdata.substr(0, pos);
  std::string value = strdata.substr(pos + 1, strdata.size() - pos);
  key  = trim(key);
  value = trim(value);
  auto tmp = extractInner(key, "'");
  if (tmp.size() > 0) {
    key = tmp;
  }
  tmp = extractInner(value, "'");
  if (tmp.size() > 0) {
    value = tmp;
  }
  return std::pair<std::string, std::string>(key, value);
}

// support only flat-1d assoc-array
std::map<std::string, std::string> parseJson(std::string jsondata) {
  std::map<std::string, std::string> jsonMap;
  std::string innerBracesData = extractInner(jsondata, "{", "}");
  // std::cerr << innerBracesData << std::endl;
  std::vector<std::string> jsonList = jsonCommaSplit(innerBracesData);
  for (auto jsonElem : jsonList) {
    auto  keyvalue = jsonKeyValueSplit(jsonElem);
      if (keyvalue.first.size() > 0) {
        jsonMap[keyvalue.first] = keyvalue.second;
      }
  }
  return jsonMap;
}

struct NPYheader_t readNPYheader(std::ifstream &fin) {
  char sig[6];
  uint16_t ver;
  uint16_t jsonlen;
  std::stringstream ss;
  NPYheader_t header;
  fin.read(sig, 6);
  if (std::memcmp(sig, "\x93NUMPY", 6) != 0) {
    ss << "wrong npy signature:" << sig;
    throw std::runtime_error(ss.str());
  }
  fin.read(reinterpret_cast<char *>(&ver), sizeof(uint16_t));
  fin.read(reinterpret_cast<char *>(&jsonlen), sizeof(uint16_t));
  if ((* reinterpret_cast<const uint16_t*>("\1\0")) != 1) {
    // native order = big endian
    ver = (ver << 8) | (ver >> 8);
    jsonlen = (jsonlen << 8) | (jsonlen >> 8);
  }
  if (0x80 < (0x0a + jsonlen)) {
    ss << "too long json length:" << jsonlen;
    throw std::runtime_error(ss.str());
  }
  std::string jsondata(jsonlen, '\0');
  if (!fin.read(reinterpret_cast<char *>(&(jsondata[0])), jsonlen)) {
    ss << "too short file for jsonlen:" << jsonlen;
    throw std::runtime_error(ss.str());
  }
  // std::cerr << jsondata << std::endl;
  // {'descr': '|u1', 'fortran_order': False, 'shape': (46, 70, 3), }
  std::map<std::string, std::string> jsonMap = parseJson(jsondata);
  for (auto itr = jsonMap.begin(); itr != jsonMap.end(); ++itr) {
    std::string key = itr->first, value = itr->second;
    if (key == "descr") {
      if ((value != "|u1") && (value != "<f4")) {
        ss << "descr:" << value << ", must be |u1 or <f4";
        throw std::runtime_error(ss.str());
      }
      header.datatype = value;
    } else if (key =="fortran_order") {
      if (value != "False") {
        throw std::runtime_error("fortran_order must be False");
      }
    } else if (key == "shape") {
      value = extractInner(value, "(", ")");
      std::vector<std::string> numstrList = jsonCommaSplit(value);
      if (numstrList.size() != 3) {
        ss << "Wrong shape size:" << numstrList.size();
        throw std::runtime_error(ss.str());
      }
      std::vector<int> shape(numstrList.size());
      for (size_t i = 0 ; i < numstrList.size() ; i++) {
        shape[i] = std::stoi(numstrList[i]);
      }
      header.shape = shape;
    } else {
      ss << "Unknown json key:" << key;
      throw std::runtime_error(ss.str());
    }
  }
  return header;
}

template<typename T>
void readNPYdata(std::ifstream &fin, const struct NPYheader_t &nh,
                 T *data) {
  if (data == NULL) {
    throw std::runtime_error("data == NULL");
  }
  int n = std::accumulate(nh.shape.begin(), nh.shape.end(), 1, std::multiplies<int>());
  // only little-endian support
  if (nh.datatype == "|u1") {
    for (int i = 0 ; i < n ; ++i) {
      int cc = fin.get();
      if (cc < 0) {
        throw std::runtime_error("incompleted rgb-data");
      }
      data[i] = cc;
    }
  } else if (nh.datatype == "<f4") {
    char fvalue_char[4];
    for (int i = 0 ; i < n ; ++i) {
      fin.read(fvalue_char, 4);
      if (fin.eof()) {
        throw std::runtime_error("incompleted rgb-data");
      }
      data[i] = *(reinterpret_cast<float*>(fvalue_char));
    }
  } else {
    throw std::runtime_error("unsupported type:"+nh.datatype);
  }
}

}  // namespace npy
}  // namespace blueoil

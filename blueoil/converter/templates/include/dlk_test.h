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
=============================================================================*/

#ifndef dlk_test_nodeS_H_INCLUDED
#define dlk_test_nodeS_H_INCLUDED


#include <iostream>
#include <string>
#include <cmath>
#include <sstream>
#include <fstream>
#include <vector>
#include "types.h"

template <class T_SIZE, class T>
struct Diff {
  Diff(T_SIZE idx, T i, T e, T d)
    : index(idx), input(i), expected(e), diff(d) {};
  T_SIZE index;
  T input;
  T expected;
  T diff;
};


namespace dlk_test
{
  template<class T>static constexpr T tolerance = 0.00001;

  template<class T_IN, class T_RES>
  inline bool same(T_IN input, T_RES expected, T_IN& diff)
  {
    auto exp = T_IN(expected);
    auto aex = std::max<T_IN>(std::abs(exp), 1);
    auto abs_diff = std::abs(input - exp);
    auto tol = tolerance<T_IN> * aex;
    if(std::is_integral<T_IN>::value)
    {
      diff = abs_diff;
      if(abs_diff == T_IN())
      {
        return true;
      }
    }
    else
    { 
      diff = abs_diff / aex;
      if(abs_diff < tol)
      {
        diff = T_IN();
        return true;
      }
    }
    return false;
  }

  template<class T>
  void dump_array(T* array, unsigned num_elems, const std::string& filepath)
  {
    std::ofstream f;
    f.open (filepath.c_str());
    for (unsigned i = 0; i < num_elems; ++i) {
      f << i << ": " << array[i];
      if (i != num_elems - 1)
        f << "\n";
      else
        f << std::endl;
    }
    f.close();
  }

  template<class T_SIZE, class T>
  void dump_diff_array(std::vector<Diff<T_SIZE,T> > array, const std::string& filepath)
  {
    std::ofstream f;
    f.open (filepath.c_str());
    unsigned num_elems = array.size();
    for (unsigned i = 0; i < num_elems; ++i) {
      Diff<T_SIZE,T>& d = array[i];
      f << "idx: " << d.index << ", in: " << d.input << ", ex: " << d.expected << ", diff: " << d.diff;
      if (i != num_elems - 1)
        f << "\n";
      else
        f << std::endl;
    }
    f.close();
  }

  template<class T_IN, class T_RES, class T_SIZE>
  inline bool compare(
    T_IN* input,
    const std::string& result_name,
    T_RES* result_array,
    T_SIZE num_elems)
  {
    int failed_count = 0;
    int failed_index = -1;

    std::vector<Diff<T_SIZE,T_IN> > diff_array;
    T_IN max_diff = 0;

    for (T_SIZE i = 0; i < num_elems; i++)
    {
      T_IN diff;
      if (!same(input[i], result_array[i], diff)) {
        if (failed_index == -1) { failed_index = i; }
        ++failed_count;
        max_diff = std::max(max_diff, diff);
        Diff<T_SIZE, T_IN> d(i, input[i], T_IN(result_array[i]), diff);
        diff_array.push_back(d);
      }
    }

    if (failed_count > 0) {
      std::cout << "-------------------------------------------------------------" << std::endl;
      std::ostringstream s_index, s_result;
      s_index << input[failed_index];
      s_result << result_array[failed_index];

      std::cout << "Comparison: " << result_name << " failed..." << "\n"
                << "Failed count: " << failed_count << "\n"
                << "Maximum diff: " << max_diff << "\n"
                << "First failed report" << "\n"
                << "index: " << failed_index << " / " << num_elems << "\n"
                << "input: " << s_index.str() << ", "
                << "expected: " << s_result.str() << "\n"
                << std::endl;

      dump_diff_array(diff_array, result_name + ".diff");

      std::cout << "-------------------------------------------------------------" << std::endl;
     } else {
       std::cout << "-------------------------------------------------------------" << std::endl;
       std::cout << "Comparison: " << result_name << " succeeded!!!" << std::endl;
       std::cout << "-------------------------------------------------------------" << std::endl;
     }

     return (failed_count < 0);
   }
} // namespace dlk_test

#endif


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

#pragma once

enum input_type
{
  SEQUENTIAL = 0,
  RANDOM = 1,
  ALL_1 = 2,
};

template <class T>
T gen_random_value(int max, int s, int b)
{
  // s*x - b such that {x | x >= 0, x < max}
  // e.g. if you need 1 or -1, you can take max = 2, s = 2, z = 1
  // e.g. if you need from 0 to 256, you can take max  = 256, s = 1, z = 0
  int x = rand() % max;
  return T((s * x) - b);
}

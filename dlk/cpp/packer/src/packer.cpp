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
==============================================================================*/

#include "packer.h"


extern "C" Packer* packer_create()
{
  return new Packer();
}

extern "C" void packer_delete(Packer *instance)
{
  if(instance != NULL)
    delete instance;
}

extern "C" void packer_set_bitwidth(Packer *packer, uint32_t bitwidth)
{
  packer->set_bitwidth(bitwidth);
}

extern "C" void packer_set_wordsize(Packer *packer, uint32_t wordsize)
{
  packer->set_wordsize(wordsize);
}

extern "C" void packer_set_multiplexing_mode(Packer *packer, uint32_t mode)
{
  packer->set_multiplexing_mode(mode);
}

extern "C" void packer_set_extra_bit_value(Packer *packer, bool value)
{
  packer->set_extra_bit_value(value);
}

extern "C" uint32_t packer_get_output_size(Packer *packer, uint32_t size, uint32_t number_of_kernels)
{
  return packer->get_output_size(size, number_of_kernels);
}

extern "C" uint32_t packer_get_wordsize(Packer *packer)
{
  return packer->get_wordsize();
}

extern "C" void packer_run(Packer *packer, float *input, uint32_t size, void *output)
{
  switch(packer->get_wordsize())
  {
    case 32:
    {
      uint32_t *output_ptr = (uint32_t *) output;
      packer->run(input, size, output_ptr);
      break;
    }
    case 8:
    {
      uint8_t *output_ptr = (uint8_t *) output;
      packer->run(input, size, output_ptr);
      break;
    }
  }
}

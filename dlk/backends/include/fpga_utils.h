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
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include "common/global.h"

namespace p = conv3x3_params;

class MappedMem
{
public:
  using memtype = volatile void;

  MappedMem(unsigned long g_paddr, uint32_t g_count, uint32_t g_size) : mem(NULL), aligned_size(0)
  {
    memtype *aligned_vaddr;
    unsigned long aligned_paddr;

    /* Align address to access size */
    g_paddr &= ~(g_size - 1);

    aligned_paddr = g_paddr & ~(4096 - 1);
    aligned_size = g_paddr - aligned_paddr + (g_count * g_size);
    aligned_size = (aligned_size + 4096 - 1) & ~(4096 - 1);

    int fd = -1;
    if ((fd = open("/dev/mem", O_RDWR, 0)) < 0)
      return;

    aligned_vaddr = mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, aligned_paddr);

    if (aligned_vaddr == NULL) {
      printf("Error mapping address %x\n", aligned_paddr);
      return;
    }

    mem = (memtype *)((uint32_t)aligned_vaddr + (uint32_t)(g_paddr - aligned_paddr));
    close(fd);
  }

  ~MappedMem()
  {
    if (mem != NULL) {
      munmap((void *)mem, aligned_size);
    }
  }

  template <typename T>
  memtype Write(T data)
  {
    T *mem_ptr = (T *)mem;
    *mem_ptr = data;
  }

  template <typename T>
  bool Check(T data)
  {
    T *mem_ptr = (T *)mem;
    return *mem_ptr == data;
  }

  template <typename T>
  memtype Read(T &data)
  {
    T *mem_ptr = (T *)mem;
    data = *mem_ptr;
  }

  template <typename T>
  memtype Write(T *data, unsigned int size)
  {
    T *mem_ptr = (T *)mem;
    for (unsigned int i = 0; i < size; i++) { *mem_ptr++ = data[i]; }
  }

  template <typename T>
  bool Check(T *data, unsigned int size)
  {
    bool success = true;
    T *mem_ptr = (T *)mem;

    for (unsigned int i = 0; i < size; i++) {
      success &= (*mem_ptr++ == data[i]);
      if (!success) {
        break;
      }
    }

    return success;
  }

  template <typename T>
  memtype Read(T *data, unsigned int size)
  {
    // volatile T* _data = data;
    T *mem_ptr = (T *)mem;
    for (unsigned int i = 0; i < size; i++) { data[i] = *mem_ptr++; }
  }

  memtype *get() { return mem; }

private:
  MappedMem();
  MappedMem(const MappedMem &);
  MappedMem &operator=(const MappedMem &);

private:
  memtype *mem;
  uint32_t aligned_size;
};

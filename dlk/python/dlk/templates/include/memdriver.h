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

#ifndef MEMORY_MAPPED_WRITE_AND_CHECK
#define MEMORY_MAPPED_WRITE_AND_CHECK

#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include <system_error>

class FDManager {
 public:
  FDManager() : fd(-1) {}
  FDManager(int fd) : fd(fd) {}
  ~FDManager() {
    if (fd >= 0) {
      close(fd);
    }
  }
  operator int() const { return fd; }
 private:
  int fd;
};

class SimpleMappedMem {
 public:
  SimpleMappedMem(std::size_t base, std::size_t size) : length(0) {
    FDManager fd(open("/dev/mem", O_RDWR | O_SYNC));
    if (fd == -1) {
      return;
    }
    int rw = PROT_READ | PROT_WRITE;
    ptr = mmap(nullptr, size, rw, MAP_SHARED, fd, base);
    if (ptr == MAP_FAILED) {
      throw std::system_error(errno, std::generic_category());
    }
    length = size;
  }
  ~SimpleMappedMem() {
    if (ptr != MAP_FAILED) {
      munmap(ptr, length);
    }
  }
  void* get() const { return ptr; }
 private:
  void* ptr;
  std::size_t length;
};

class MappedMem
{
public:
  using memtype = volatile void;
  MappedMem() = delete;
  MappedMem(const MappedMem &) = delete;
  MappedMem& operator=(const MappedMem &) = delete;

  MappedMem(unsigned long g_paddr,
            uint32_t g_count,
            uint32_t g_size)
    : mem(nullptr), aligned_size(0)
  {
    memtype* aligned_vaddr;
    unsigned long aligned_paddr;

    /* Align address to access size */
    g_paddr &= ~(g_size - 1);

    aligned_paddr = g_paddr & ~(4096 - 1);
    aligned_size = g_paddr - aligned_paddr + (g_count * g_size);
    aligned_size = (aligned_size + 4096 - 1) & ~(4096 - 1);

    FDManager fd(open("/dev/mem", O_RDWR));
    if (fd == -1) {
      return;
    }

    aligned_vaddr = mmap(nullptr,
                         aligned_size,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED,
                         fd, aligned_paddr);

    if (aligned_vaddr == MAP_FAILED) {
      printf("Error mapping address %lx\n", aligned_paddr);
      return;
    }

    mem = (memtype *)((uint32_t)aligned_vaddr + (uint32_t)(g_paddr - aligned_paddr));
  }

  ~MappedMem()
  {
    if(mem != nullptr)
      munmap((void*)mem, aligned_size);
  }


  template<typename T>
  memtype Write(T data)
  {
    T *mem_ptr = (T *) mem;
    *mem_ptr = data;
  }

  template<typename T>
  bool Check(T data)
  {
    T *mem_ptr = (T *) mem;
    return *mem_ptr == data;
  }


  template<typename T>
  memtype Read(T &data)
  {
    T *mem_ptr = (T *) mem;
    data = *mem_ptr;
  }


  template<typename T>
  memtype Write(const T *data, unsigned int size)
  {
    T *mem_ptr = (T *) mem;
    for(unsigned int i = 0; i < size; i++)
      *mem_ptr++ = data[i];
  }


  template<typename T>
  bool Check(const T *data, unsigned int size)
  {
    bool success = true;
    T *mem_ptr = (T *) mem;

    for(unsigned int i = 0; i < size; i++)
    {
      success &= (*mem_ptr++ == data[i]);
      if(!success)
        break;
    }

    return success;
  }


  template<typename T>
  memtype Read(T *data, unsigned int size)
  {
    // volatile T* _data = data;
    T *mem_ptr = (T *) mem;
    for(unsigned int i = 0; i < size; i++)
      data[i] = *mem_ptr++;
  }


  memtype* get()
  {
    return mem;
  }

private:
  memtype *mem;
  uint32_t aligned_size;
};


#endif

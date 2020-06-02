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
#include <sys/mman.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <map>
#include <vector>

class DMA_Buffer
{

public:

  DMA_Buffer()
    :
    mm_buffer(nullptr),
    mapped_size_in_bytes(0)
  {}


  ~DMA_Buffer()
  {
    if(mm_buffer != nullptr)
    {
      munmap((void *) mm_buffer, mapped_size_in_bytes);

      if(using_dma_cache)
      {
        for(auto p : ctrl)
          close(p.second);
      }
    }
  }


  bool init(const std::string &device_name, uint32_t elements, uint32_t element_size, bool use_dma_cache, unsigned long physical_address)
  {
    if(mm_buffer != nullptr)
    {
      std::cout << "Error: DMA buffer " << device_name << " already initalized" << std::endl;
      return false;
    }

    const std::string device_file = "/dev/" + device_name;
    using_dma_cache = use_dma_cache;

    if(using_dma_cache)
    {
        const std::string sys_class_device_directory = "/sys/class/udmabuf/" + device_name;

        // read only control attributes
        for(auto name : ro_ctrl_names)
        {
            std::string fname = sys_class_device_directory + "/" + name;
            ctrl[name] = open(fname.c_str(), O_RDONLY);
        }

        // read/write control attributes
        for(auto name : rw_ctrl_names)
        {
            std::string fname = sys_class_device_directory + "/" + name;
            ctrl[name] = open(fname.c_str(), O_WRONLY);
        }

        // constant value, read only one time
        ssize_t bytes_read = read(ctrl["phys_addr"], attribute_buffer, 1024);
        if(bytes_read == 0)
          return false;

        int match = sscanf(attribute_buffer, "%lx", &phys_addr);
        if(match != 1)
          return false;
    }
    else
    {
      phys_addr = physical_address;
    }

    // open device and map the memory

    int dev_fd;

    if(using_dma_cache)
    {
        dev_fd = open(device_file.c_str(), O_RDWR);
    }
    else
    {
        dev_fd = open(device_file.c_str(), O_RDWR | O_SYNC);
    }

    if(dev_fd < 0)
    {
      std::cout << strerror(errno) << std::endl;
      return false;
    }

    mapped_size_in_bytes = elements * element_size;
    mm_buffer = mmap(
      nullptr,
      mapped_size_in_bytes,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      dev_fd,
      physical_address
    );

    close(dev_fd);
    if(mm_buffer == MAP_FAILED)
    {
      std::cout << "Error: " << strerror(errno) << std::endl;
      return false;
    }

    return true;
  }

  unsigned long physical_address()
  {
    return phys_addr;
  }

  bool sync_for_cpu()
  {
    if(using_dma_cache)
    {
      ssize_t written = write(ctrl["sync_for_cpu"], "1", 1);
      return written == strlen("1");
    }
    else
      return true;
  }

  bool sync_for_device()
  {
    if(using_dma_cache) {
      ssize_t written = write(ctrl["sync_for_device"], "1", 1);
      return written == strlen("1");
    }
    else
      return true;
  }

  bool sync_size(unsigned long size)
  {
    if(using_dma_cache)
    {
      sprintf(attribute_buffer, "%lu", size);
      ssize_t len = strlen(attribute_buffer);
      ssize_t written = write(ctrl["sync_size"], attribute_buffer, len);
      return written == len;
    }

    return true;
  }

  bool sync_offset(unsigned long offset)
  {
    if(using_dma_cache)
    {
      sprintf(attribute_buffer, "%lu", offset);
      ssize_t len = strlen(attribute_buffer);
      ssize_t written = write(ctrl["sync_offset"], attribute_buffer, len);
      return written == len;
    }

    return true;
  }

  volatile void* buffer()
  {
    return mm_buffer;
  }


private:
  DMA_Buffer(const DMA_Buffer &);
  DMA_Buffer& operator=(const DMA_Buffer &);

private:
  volatile void *mm_buffer;
  uint32_t mapped_size_in_bytes;
  unsigned long phys_addr;
  char attribute_buffer[1024];
  bool using_dma_cache;

  const std::vector<std::string> ro_ctrl_names = {"phys_addr", "size"};
  const std::vector<std::string> rw_ctrl_names = {"sync_for_cpu", "sync_for_device", "sync_offset", "sync_size"};

  std::map<std::string, int> ctrl;
};

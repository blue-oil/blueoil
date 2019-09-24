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

class FileDescriptor {
 public:
  FileDescriptor() : fd(-1) {}
  FileDescriptor(int fd) : fd(fd) {}
  ~FileDescriptor() {
    if (fd >= 0) {
      close(fd);
    }
  }
  operator int() const { return fd; }
 private:
  int fd;
};

class MappedMem {
 public:
  MappedMem(std::size_t base, std::size_t size) : length(0) {
    FileDescriptor fd(open("/dev/mem", O_RDWR | O_SYNC));
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
  ~MappedMem() {
    if (ptr != MAP_FAILED) {
      munmap(ptr, length);
    }
  }
  void* get() const { return ptr; }
 private:
  void* ptr;
  std::size_t length;
};

#endif

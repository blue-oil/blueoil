#pragma once

#include "memdriver.h"

class FPGAComponent
{

public:

  FPGAComponent(uint32_t csr_base_address)
    : start_reg (csr_base_address, 1, sizeof(uint32_t)),
      done_reg  (csr_base_address + 0x10, 1, sizeof(uint32_t)),
      mem_reg (csr_base_address + 0x18, 1, sizeof(uint32_t)),
      size_reg  (csr_base_address + 0x20, 1, sizeof(uint32_t))
  {}

  void setup(uint32_t mem_address, uint32_t size)
  {
    mem_reg.Write(mem_address);
    size_reg.Write(size);
  }

  void run()
  {
    start_reg.Write(0x1);
  }

  void wait()
  {
    volatile uint32_t done_flag = 0;
    while (!(done_flag & 0x2)) {
      done_reg.Read(done_flag);
    }
  }

private:

  MappedMem start_reg;
  MappedMem done_reg;
  MappedMem mem_reg;
  MappedMem size_reg;
};

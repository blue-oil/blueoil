#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <bitset>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "memdriver.h"

#define CLOCK_TYPE CLOCK_THREAD_CPUTIME_ID

#define MGR_STAT_ADDR 0xFF706000
#define MGR_STAT_OFFSET 0x0

#define MGR_CTRL_ADDR 0xFF706000
#define MGR_CTRL_OFFSET 0x4

#define GPIO_PORTA_EOI_ADDR 0xFF706000
#define GPIO_PORTA_EOI_OFFSET 0x84C

#define MGR_DATA_ADDR 0xFFB90000
#define MGR_DATA_OFFSET 0x0

#define MGR_DCLCK_STAT_ADDR 0xFF706000
#define MGR_DCLCK_STAT_OFFSET 0xC

#define MGR_DCLCK_COUNT_ADDR 0xFF706000
#define MGR_DCLCK_COUNT_OFFSET 0x8

#define MGR_CTRL_CDRATIO_8_MASK_UP 0x000000C0
#define MGR_CTRL_CFGWDTH_32_MASK_UP 0x00000200
#define MGR_CTRL_NCE_0_MASK_DOWN ~0x00000002
#define MGR_CTRL_EN_HPSCONF_MASK_UP 0x00000001
#define MGR_CTRL_NCONFIGPULL_MASK_UP 0x00000004
#define MGR_CTRL_NCONFIGPULL_MASK_DOWN ~0x00000004
#define MGR_STAT_MODE_BITS 0x00000007
#define MGR_GPIO_PORTA_EOI_NS_CLEAR 0x00000001
#define MGR_CTRL_AXICFGEN_ENABLE 0x00000100
#define MGR_CTRL_AXICFGEN_DISABLE ~0x00000100
#define MGR_CTRL_EN_HPSCONF_MASK_DOWN ~0x00000001

#define GPIO_EXT_PORTA_ADDR 0xFF706000
#define GPIO_EXT_PORTA_OFFSET 0x850


char* load_rbf(const char *filename, size_t *bytes_read)
{
  FILE *f = NULL;
  f = fopen(filename, "rb");
  if(f == NULL) {
    fprintf(stderr, "Error: cannot open file %s\n", filename);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  size_t file_size = ftell (f);
  rewind(f);

  // load into buffer
  char *buf = (char*) malloc(sizeof(char) * file_size + sizeof(uint32_t));
  *bytes_read = fread(buf, sizeof(char), file_size, f);
  fclose(f);

  if (*bytes_read != file_size) {
    fprintf(stderr, "Error while reading file %s (bytes read %d of %d)\n", filename, *bytes_read, file_size);
    return NULL;
  }

  return buf;
}



int main(int argc, const char* argv[])
{
  if(argc != 2) {
    std::cout << "Use: " << argv[0] << " <.rbf file>" << std::endl;
    return 0;
  }

  // read the rbf
  size_t bytes_read = 0;
  const uint32_t *config_words =  (uint32_t *) load_rbf(argv[1], &bytes_read);
  if(config_words == NULL) {
    std::cout << "Error reading .rbf file" << std::endl;
    return 0;
  }

  size_t word_size = sizeof(uint32_t);
  uint32_t aligned_size(0);

  MappedMem mgr_stat_reg(MGR_STAT_ADDR + MGR_STAT_OFFSET, 1, sizeof(uint32_t));
  MappedMem mgr_ctrl_reg(MGR_CTRL_ADDR + MGR_CTRL_OFFSET, 1, sizeof(uint32_t));
  MappedMem gpio_porta_eoi_reg(GPIO_PORTA_EOI_ADDR + GPIO_PORTA_EOI_OFFSET, 1, sizeof(uint32_t));
  MappedMem mgr_data_reg(MGR_DATA_ADDR + MGR_DATA_OFFSET, 1, sizeof(uint32_t));
  MappedMem gpio_ext_porta_reg(GPIO_EXT_PORTA_ADDR + GPIO_EXT_PORTA_OFFSET, 1, sizeof(uint32_t));
  MappedMem dclk_stat_reg(MGR_DCLCK_STAT_ADDR + MGR_DCLCK_STAT_OFFSET, 1, sizeof(uint32_t));
  MappedMem dclk_cnt_reg(MGR_DCLCK_COUNT_ADDR + MGR_DCLCK_COUNT_OFFSET, 1, sizeof(uint32_t));

  // timming starts here
  struct timespec ts;
  clock_gettime (CLOCK_TYPE, &ts);
  uint64_t start = uint64_t(ts.tv_sec * 1000000000 + ts.tv_nsec);

  // stat
  volatile uint32_t stat_reg = 0;
  mgr_stat_reg.Read(stat_reg);

  // ctrl
  volatile uint32_t ctrl_reg = 0, cfg_ctrl_reg = 0;
  mgr_ctrl_reg.Read(ctrl_reg);

  cfg_ctrl_reg = ctrl_reg;

  cfg_ctrl_reg |= MGR_CTRL_CDRATIO_8_MASK_UP;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  cfg_ctrl_reg |= MGR_CTRL_CFGWDTH_32_MASK_UP;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  cfg_ctrl_reg &= MGR_CTRL_NCE_0_MASK_DOWN;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  cfg_ctrl_reg |= MGR_CTRL_EN_HPSCONF_MASK_UP;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  cfg_ctrl_reg |= MGR_CTRL_NCONFIGPULL_MASK_UP;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  mgr_stat_reg.Read(stat_reg);

  while((stat_reg & MGR_STAT_MODE_BITS) != 0x1)
    mgr_stat_reg.Read(stat_reg);

  std::cout << "In reset phase!" << std::endl;

  cfg_ctrl_reg &= MGR_CTRL_NCONFIGPULL_MASK_DOWN;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  mgr_stat_reg.Read(stat_reg);

  mgr_stat_reg.Read(stat_reg);
  while(stat_reg & MGR_STAT_MODE_BITS != 0x2)
    mgr_stat_reg.Read(stat_reg);

  std::cout << "In configuration phase!" << std::endl << std::endl;

  volatile uint32_t gpio_ns_clear = MGR_GPIO_PORTA_EOI_NS_CLEAR;
  gpio_porta_eoi_reg.Write(gpio_ns_clear);

  mgr_ctrl_reg.Read(ctrl_reg);
  cfg_ctrl_reg = ctrl_reg;

  cfg_ctrl_reg |= MGR_CTRL_AXICFGEN_ENABLE;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  std::cout << "In configuration phase: ns clear and axcfgen" << std::endl << std::endl;

  size_t i = 0;
  volatile uint32_t dummy_word = 0x00000123;

  std::cout << "Writing configuration: " << bytes_read / 4 << " words" << std::endl;
  while(bytes_read >= word_size)
  {
    mgr_data_reg.Write(config_words[i++] + 0);
    bytes_read -= word_size;
  }

  volatile uint32_t sw = config_words[i++];
  if(bytes_read == 3)
    mgr_data_reg.Write(sw & 0x00ffffff);
  else if(bytes_read == 2)
    mgr_data_reg.Write(sw & 0x0000ffff);
  else if(bytes_read == 1)
    mgr_data_reg.Write(sw & 0x000000ff);
  else
    ;

  std::cout << "Waiting for conf done and status" << std::endl << std::endl;

  volatile uint32_t ext_porta_reg = 0;
  gpio_ext_porta_reg.Read(ext_porta_reg);
  while((ext_porta_reg & 0x3) != 0x3)
    gpio_ext_porta_reg.Read(ext_porta_reg);

  mgr_ctrl_reg.Read(ctrl_reg);
  cfg_ctrl_reg = ctrl_reg;

  cfg_ctrl_reg &= MGR_CTRL_AXICFGEN_DISABLE;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  volatile uint32_t dclck_done = 0x1;
  dclk_stat_reg.Write(dclck_done);

  volatile uint32_t dclk_cnt = 0x4;
  dclk_cnt_reg.Write(dclk_cnt);

  dclck_done = 0x0;
  dclk_stat_reg.Read(dclck_done);
  while((dclck_done & 0x1) != 0x1)
    dclk_stat_reg.Read(dclck_done);

  dclck_done = 0x1;
  dclk_stat_reg.Write(dclck_done);

  mgr_stat_reg.Read(stat_reg);
  while(stat_reg & MGR_STAT_MODE_BITS != 0x4)
    mgr_stat_reg.Read(stat_reg);

  std::cout << "In user mode state!" << std::endl;
  std::cout << "Stat: " << std::bitset<32>(stat_reg) << std::endl << std::endl;

  mgr_ctrl_reg.Read(ctrl_reg);
  cfg_ctrl_reg = ctrl_reg;
  cfg_ctrl_reg &= MGR_CTRL_EN_HPSCONF_MASK_DOWN;
  mgr_ctrl_reg.Write(cfg_ctrl_reg);

  std::cout << "Configuration finished" << std::endl;

  clock_gettime (CLOCK_TYPE, &ts);

  // Show time
  uint64_t elapsed_time_us = (uint64_t(ts.tv_sec * 1000000000 + ts.tv_nsec) - start) / 1000;
  std::cout << "Time: " << elapsed_time_us << " us" << ", " << start << std::endl;

  return 0;
}

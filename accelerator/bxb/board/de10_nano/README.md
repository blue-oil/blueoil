# Prerequisites

- [sbt](https://www.scala-sbt.org/)
- Intel [Quartus](https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/overview.html)
  - Tested with 18.0
- Intel [SoC EDS](https://www.intel.com/content/www/us/en/software/programmable/soc-eds/overview.html)
  - Tested with 18.0

# Building preloader and rbf

**prepare environment to have access to both fpga and soc eds tools**
```
$ export INTEL_FPGA_TOOLS=<path_to_quartus_installation>
$ export PATH=$PATH:$INTEL_FPGA_TOOLS/quartus/bin:$INTEL_FPGA_TOOLS/modelsim_ase/bin:$INTEL_FPGA_TOOLS/quartus/sopc_builder/bin
$ <path_to_soc_eds_installation>/embedded/embedded_command_shell.sh
```

**build bitstream and preloader**
```
$ cd <path_to_bxb_project>/board/de10_nano/
$ make
```
Upon successfull completion copy `software/spl_bsp/preloader-mkpimage.bin` and `output_files/soc_system.rbf` from the directory.

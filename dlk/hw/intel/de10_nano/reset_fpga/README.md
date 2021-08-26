# Cyclone V / DE10Nano programmer

## Compilation
```
$ cd dlk/hw/intel/de10_nano/reset_fpga
$ vim CMakeLists.txt  # select the ARM toolchain you want to use (from reset_fpga/toolchain/ directory)
$ mkdir build
$ cd build/
$ cmake .. && make
```

A new binary called `progfpga` is created in `reset_fpga/build/` directory.

## How to use

1. Copy the `progfpga` binary to the FPGA
2. Run the binary passing the `.rbf` file as a command line argument:
```
root@de10nano:~$ ./progfpga ./configuration_file.rbf
```

## Important
* If the port configuration of the FPGA design change then it is necessary to update the preloader image and reboot the board.
* If the FPGA design does not change then is safe to re-program the FPGA with this tool. In this situation, reboot or replace the preloader is NOT required.


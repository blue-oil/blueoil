# How to Generate rootfs for 32 and 64 bit ARM

This document explains scripts to generate rootfs for ARM CPU and how to use those scripts.

## Confirmed Environment:
- Ubuntu 16.04.4 LTS (Xenial Xerus)
- Docker 18.09.0
- bash

## Prerequisite
You need to install QEMU user mode emulation binaries.
```
apt install qemu-user-static
```

## How to run the script

The operation is defferent between 32bit and 64bit.

In case you want to make 32bit rootfs, please perfome commands below:

```
# For 32bit
cd blueoil
make rootfs-armhf
# You can see the generated rootfs at blueoil/make_os/build/rootfs_armhf.tgz
```

In case you want to make 64bit rootfs, please perfome commands below:

```
# For 64bit
cd blueoil
make rootfs-arm64
# You can see the generated rootfs at blueoil/make_os/build/rootfs_arm64.tgz
```

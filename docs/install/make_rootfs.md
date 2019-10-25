# How to Generate rootfs for 32 and 64 bit ARM

This document explains scripts to generate rootfs for ARM CPU and how to use those scripts.

## Confirmed Environment:
- Ubuntu 16.04.4 LTS (Xenial Xerus)
- Docker 18.09.0
- bash

## How to run the script

Making docker is necessary before you actually run the command to generate rootfs.

```
make os-docker
```

After this, the operation is defferent between 32bit and 64bit.

In case you want to make 32bit rootfs, please perfome commands below:

```
# For 32bit
cd blueoil
make rootfs32
# You can see the generated rootfs at blueoil/dlk/hw/make_os/build/rootfs32.tgz
```

In case you want to make 64bit rootfs, please perfome commands below:

```
# For 64bit
cd blueoil
make rootfs64
# You can see the generated rootfs at blueoil/dlk/hw/make_os/build/rootfs64.tgz
```


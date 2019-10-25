# How to Generate rootfs for 32 and 64 bit ARM

Caution: This document should be merged into the formal other documents soon after scripts for rootfs has been merged into master.

This is document of scripts to generate rootfs for ARM CPU.
This document also describes how to use those scripts.

## Confirmed Environment:
- Ubuntu 16.04.4 LTS (Xenial Xerus)
- Docker 18.09.0
- bash

## How to run the script

```
# For 32bit
cd blueoil/dlk/hw/make_os
./docker_run_rootfs32.sh
# You can see the generated rootfs at blueoil/dlk/hw/make_os/build/rootfs32.tgz

# For 64bit
cd blueoil/dlk/hw/make_os
./docker_run_rootfs64.sh
# You can see the generated rootfs at blueoil/dlk/hw/make_os/build/rootfs64.tgz
```


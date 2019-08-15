## Connection

### Cannot connect to the FPGA board via serial
Do the following steps if the FPGA board cannot be connected via serial, or a `Line in use` message shows up in the terminal. 
1. Disconnect the FPGA board from your PC.
2. Reboot the FPGA board and reconnect it to your PC.
3. If step 1-2. doesn't help, do step 1., reboot your PC, and do step 2. again.

### Cannot connect to the FPGA board via ssh
You need ethernet LAN connection with the FPGA board to connect to it via ssh.
The IP address of your FPGA board can be found using the `ifconfig` command.

## OS

### The Linux OS does not boot
If the Linux OS does not boot, it is likely that your OS is not properly written. 
Please erase your microSD card and write the OS again.
We recommend using [Etcher](https://www.balena.io/etcher/) to help you write the image on any platform.
Please refer to the details in [Download Linux system image (On your PC)](../install/install.html#download-linux-system-image-on-your-pc).

## Errors

### `memory error` in python
It is likely that you don't have enough disk space. 
You can check it using the `fdisk` command. 
If you did not expand your root partition, we recommend doing it by following the final part of [Connect to the FPGA board (On your PC)](../install/install.html#connect-to-the-fpga-board-on-your-pc).

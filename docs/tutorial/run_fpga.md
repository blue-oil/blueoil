# Run trained neural network on the FPGA board

Blueoil prepared demonstration scripts to showcase the examples of classification and object detection
using a DE10-Nano Kit board and USB camera.

#### Setup

- The DE10-Nano: Prepare the board and create Linux system on microSD card. (Please see the detail in <a href="../install/install.html">Installation</a>)
- USB camera: After setting up the DE10-Nano board, connect the USB camera to De10-Nano board.
Make sure the camera is recognized by the device.

#### Preparation

- From the Setup step, you should be able to login to the DE10-Nano
through ssh inside an xterm program.

      $ ssh -X root@{DE10-Nano's IP}

- Preparing the `/demo` (generated as `output` by `blueoil convert`) directory contains the following on the DE10-Nano:

```
demo
 ├── fpga
 │   └── soc_system.rbf
 ├── models
 │   ├── lib
 │   │   └── lib_fpga.so
 │   └── meta.yaml
 └── python
     ├── lmnet
     ├── requirements.txt
     └── usb_camera_demo.py
```

#### Update FPGA configuration
Explore into the `demo/fpga` directory, and copy `soc_system.rbf` to boot partition (/dev/mmcblk0p1).

      $ cd demo/fpga
      $ sudo mount /dev/mmcblk0p1 /media
      $ cp soc_system.rbf /media
      $ reboot

#### Run the demonstration
Explore into the `demo/python` directory, and execute the following command on the device.

    $ cd demo/python
    $ pip install -r requirements.txt
    $ python usb_camera_demo.py \
          -m ../models/lib/lib_fpga.so \
          -c ../models/meta.yaml

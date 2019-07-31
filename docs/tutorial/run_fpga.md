# Run a trained neural network on the FPGA board

The Blueoil repository contains pre-prepared demonstration scripts to showcase examples of classification and object detection
using a DE10-Nano Kit board and a USB camera.

#### Setup

- The DE10-Nano: Prepare the board and create a Linux system on a microSD card. (For details  please see <a href="../install/install.html">Installation</a>.)
- USB camera: After setting up the DE10-Nano board, connect the USB camera to the DE10-Nano board.
Make sure the camera is recognized by the device.

#### Preparation

- After having done the Setup step, you should be able to login to the DE10-Nano via ssh.

      $ ssh -X root@{DE10-Nano's IP}

- Copy the directory generated as `output` by `blueoil convert` to the DE10-Nano as `/demo`. It should contain the following:

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
Explore into the `demo/fpga` directory, and copy `soc_system.rbf` to the boot partition (/dev/mmcblk0p1).

      $ cd demo/fpga
      $ sudo mount /dev/mmcblk0p1 /media
      $ cp soc_system.rbf /media
      $ reboot

#### Run the demonstration
Explore into the `demo/python` directory, and execute the following commands on the device.

    $ cd demo/python
    $ pip install -r requirements.txt
    $ python usb_camera_demo.py \
          -m ../models/lib/lib_fpga.so \
          -c ../models/meta.yaml

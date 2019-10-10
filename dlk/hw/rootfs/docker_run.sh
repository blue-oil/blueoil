docker run \
 -v /storage2/Users/takeda/work_bxb_20190815/u-boot-xlnx/:/home/test/u-boot-xlnx \
 -v /storage2/Users/takeda/work_bxb_20190815/atf/arm-trusted-firmware:/home/test/arm-trusted-firmware \
 -v /storage2/Users/takeda/work_bxb_20190815/linux-xlnx/:/home/test/linux-xlnx \
 -v /storage2/Users/takeda/work_bxb_20190815/ubuntu_rootfs/:/home/test/ubuntu_rootfs \
 -v /storage2/Users/takeda/work_bxb_20190815/udmabuf/:/home/test/udmabuf \
 -it --privileged ubuntu1804

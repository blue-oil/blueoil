#!/bin/bash

# Add test user
#RUN useradd test -u 1010
#RUN echo 'test:testpass' | chpasswd
#RUN echo "test  ALL=(ALL)  ALL" >> /etc/sudoers

# Make rootfs
debootstrap --arch=arm64 --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg --verbose --foreign bionic /build/rootfs
cp /usr/bin/qemu-aarch64-static /build/rootfs/usr/bin/
#COPY ./setting_after_chroot.sh /build/rootfs
cp /build/setting_after_chroot.sh /build/rootfs/
chroot /build/rootfs /bin/bash /setting_after_chroot.sh


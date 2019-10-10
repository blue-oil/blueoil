#!/bin/bash

sudo debootstrap --arch=arm64 --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg --verbose --foreign bionic rootfs
sudo apt-get update
sudo apt-get install qemu-user-static
sudo apt-get install vim
sudo cp /usr/bin/qemu-aarch64-static rootfs/usr/bin/
sudo cp make_ubuntu_2nd.sh rootfs/
sudo chroot rootfs/ /bin/bash


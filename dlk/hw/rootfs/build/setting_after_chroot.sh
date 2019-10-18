#!/bin/bash

/debootstrap/debootstrap --second-stage # After chroot
echo "ubuntu" > /etc/hostname # After chroot
useradd ubuntu --create-home --shell /bin/bash # After chroot
echo 'ubuntu:ubuntu' | chpasswd # After chroot
echo "ubuntu  ALL=(ALL)  ALL" >> /etc/sudoers # This enable user to sudo

# TODO

#Install Network tools (ifup, ifdown, Network Manager...)
#Copy .bashrc # Is this really necessary ?

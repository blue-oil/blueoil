#!/bin/bash

# This script should be run after chroot
/debootstrap/debootstrap --second-stage
echo "ubuntu" > /etc/hostname
useradd ubuntu --create-home --shell /bin/bash
echo 'ubuntu:ubuntu' | chpasswd
echo "ubuntu  ALL=(ALL)  ALL" >> /etc/sudoers # This enables user to sudo

# TODO: Install Network tools (ifup, ifdown, Network Manager...)

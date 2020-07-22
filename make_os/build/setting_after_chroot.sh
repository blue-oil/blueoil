#!/bin/bash -ex

# This script should be run after chroot
/debootstrap/debootstrap --second-stage
echo "ubuntu" > /etc/hostname
useradd ubuntu --create-home --shell /bin/bash
echo 'ubuntu:ubuntu' | chpasswd
echo "ubuntu  ALL=(ALL)  ALL" >> /etc/sudoers # This enables user to sudo

# Install minimum packages
apt update
apt install -y --no-install-recommends openssh-server parted

# Network settings
cat << EOF > /etc/netplan/99-network-config.yaml
network:
    ethernets:
        eth0:
            dhcp4: true
    version: 2
EOF

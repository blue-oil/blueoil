#!/bin/bash

if [ $# -ne 1 ]; then
	echo "Error: Number of argument should be 1"
	exit 0
fi

BIT_32_OR_64=$1

if [ $1 = 32 ]; then
	DEBOOTSTRAP_ARCH=armhf
	QEMU_ARCH=arm
elif [ $1 = 64 ]; then
	DEBOOTSTRAP_ARCH=arm64
	QEMU_ARCH=aarch64
else
	echo "Error: Argumnet should be 32 or 64"
	exit 0
fi

BUILD_DIR=/build
ROOTFS_DIR=$BUILD_DIR/rootfs

# Chcek if the $ROOTFS_DIR exists or not
if [ -d $ROOTFS_DIR ]; then
	echo "Error: $ROOTFS_DIR already exists"
	exit 0
fi

# Check if $ROOTFS_DIR/rootfs.tgz exists or not
if [ -e $BUILD_DIR/rootfs.tgz ]; then
	echo "Error: $BUILD_DIR/rootfs.tgz already exists"
	exit 0
fi

# Add test user
#RUN useradd test -u 1010
#RUN echo 'test:testpass' | chpasswd
#RUN echo "test  ALL=(ALL)  ALL" >> /etc/sudoers

# Make rootfs
debootstrap --arch=arm64 --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg --verbose --foreign bionic $ROOTFS_DIR
cp /usr/bin/qemu-aarch64-static $ROOTFS_DIR/usr/bin/
#COPY ./setting_after_chroot.sh /build/rootfs
cp $BUILD_DIR/setting_after_chroot.sh $ROOTFS_DIR
chroot $ROOTFS_DIR /bin/bash /setting_after_chroot.sh

# Have Returned from chroot
rm $ROOTFS_DIR/setting_after_chroot.sh
tar cvzf $BUILD_DIR/rootfs.tgz -C $BUILD_DIR rootfs --remove-file
chmod a=rw $BUILD_DIR/rootfs.tgz


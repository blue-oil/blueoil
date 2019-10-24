#!/bin/bash

if [ $# -ne 1 ]; then
	echo "Error: Number of argument should be 1"
	exit 0
fi

BIT_32_OR_64=$1

if [ $1 = 32 ]; then
	DEBOOTSTRAP_ARCH=armhf
	QEMU_ARCH=arm
	OUTPUT_FNAME=rootfs32.tgz
elif [ $1 = 64 ]; then
	DEBOOTSTRAP_ARCH=arm64
	QEMU_ARCH=aarch64
	OUTPUT_FNAME=rootfs64.tgz
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

# Check if $ROOTFS_DIR/$OUTPUT_NAME exists or not
if [ -e $BUILD_DIR/$OUTPUT_FNAME ]; then
	echo "Error: $BUILD_DIR/$OUTPUT_FNAME already exists"
	exit 0
fi

# Make rootfs
debootstrap --arch=$DEBOOTSTRAP_ARCH --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg --verbose --foreign bionic $ROOTFS_DIR
cp /usr/bin/qemu-${QEMU_ARCH}-static $ROOTFS_DIR/usr/bin/
cp $BUILD_DIR/setting_after_chroot.sh $ROOTFS_DIR
chroot $ROOTFS_DIR /bin/bash /setting_after_chroot.sh

# Here is after chroot
rm $ROOTFS_DIR/setting_after_chroot.sh
tar cvzf $BUILD_DIR/$OUTPUT_FNAME -C $BUILD_DIR rootfs --remove-file
chmod a=rw $BUILD_DIR/$OUTPUT_FNAME


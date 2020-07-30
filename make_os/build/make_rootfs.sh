#!/bin/bash -ex

if [ $# -ne 1 ]; then
	echo "Error: Number of argument should be 1"
	exit 1 
fi

DEBOOTSTRAP_ARCH=$1

if [ "${DEBOOTSTRAP_ARCH}" = "armhf" ]; then
	QEMU_ARCH=arm
elif [ "${DEBOOTSTRAP_ARCH}" = "arm64" ]; then
	QEMU_ARCH=aarch64
else
	echo "Error: Argument should be armhf or arm64"
	exit 1
fi

OUTPUT_FNAME=rootfs_${DEBOOTSTRAP_ARCH}.tgz

BUILD_DIR=/build
ROOTFS_DIR=${BUILD_DIR}/rootfs

# Chcek if the $ROOTFS_DIR exists or not
if [ -d "${ROOTFS_DIR}" ]; then
	echo "Error: ${ROOTFS_DIR} on docker environment already exists"
	exit 1
fi

# Check if $ROOTFS_DIR/$OUTPUT_NAME exists or not
if [ -e "${BUILD_DIR}/${OUTPUT_FNAME}" ]; then
	echo "Error: ${BUILD_DIR}/${OUTPUT_FNAME} on docker environment already exists"
	exit 1
fi

# Make rootfs
debootstrap --arch=${DEBOOTSTRAP_ARCH} --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg --verbose --foreign bionic ${ROOTFS_DIR}
cp /usr/bin/qemu-${QEMU_ARCH}-static ${ROOTFS_DIR}/usr/bin/
cp ${BUILD_DIR}/setting_after_chroot.sh ${ROOTFS_DIR}
chroot ${ROOTFS_DIR} /bin/bash /setting_after_chroot.sh

# Here is after chroot
rm ${ROOTFS_DIR}/setting_after_chroot.sh
tar czf ${BUILD_DIR}/${OUTPUT_FNAME} -C ${BUILD_DIR} rootfs --remove-file
chmod a=rw ${BUILD_DIR}/${OUTPUT_FNAME}

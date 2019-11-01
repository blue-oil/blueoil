FROM ubuntu:18.04

RUN apt-get update && apt-get install -y debootstrap \
qemu-user-static \
&& rm -rf /var/lib/apt/lists/*

# This are necessary package for compiling kernel for future so that they're remained as comment.
#
#gcc-8-aarch64-linux-gnu \
#vim \
#make \
#sudo \
#gcc \
#bison \
#flex \
#device-tree-compiler \

RUN ln -s /usr/bin/aarch64-linux-gnu-gcc-8 /usr/local/bin/aarch64-linux-gnu-gcc && \
ln -s /usr/bin/aarch64-linux-gnu-cpp-8 /usr/local/bin/aarch64-linux-gnu-cpp


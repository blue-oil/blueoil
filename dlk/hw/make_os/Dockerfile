FROM ubuntu:18.04

RUN apt-get update && apt-get install -y gcc-8-aarch64-linux-gnu \
make \
sudo \
gcc \
bison \
flex \
device-tree-compiler \
debootstrap \
qemu-user-static \
vim \
&& rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/aarch64-linux-gnu-gcc-8 /usr/local/bin/aarch64-linux-gnu-gcc && \
ln -s /usr/bin/aarch64-linux-gnu-cpp-8 /usr/local/bin/aarch64-linux-gnu-cpp


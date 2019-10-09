FROM ubuntu:18.04

#RUN apt-get update && apt-get install -y gcc-7-aarch64-linux-gnu
RUN apt-get update && apt-get install -y gcc-8-aarch64-linux-gnu
RUN apt-get install -y make
RUN apt-get install -y sudo
RUN apt-get install -y gcc
RUN apt-get install -y bison
RUN apt-get install -y flex
RUN apt-get install -y device-tree-compiler

# These are for ROOTFS of ubuntu
RUN apt-get install -y debootstrap
#RUN apt-get install -y vim

# Add test user
RUN useradd test -u 1010
RUN echo 'test:testpass' | chpasswd
RUN echo "test  ALL=(ALL)  ALL" >> /etc/sudoers

#RUN ln -s /usr/bin/aarch64-linux-gnu-gcc-7 /usr/local/bin/aarch64-linux-gnu-gcc 
RUN ln -s /usr/bin/aarch64-linux-gnu-gcc-8 /usr/local/bin/aarch64-linux-gnu-gcc
RUN ln -s /usr/bin/aarch64-linux-gnu-cpp-8 /usr/local/bin/aarch64-linux-gnu-cpp

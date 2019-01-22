# Building and Running the image of docker for DLK

The docker is used for executing DLK to reduce the time to setup.
This document shows how to build and run the docker image for DLK.
The basic operation of docker is totally same as general use.
We show the special operation for DLK in this document.

## Build the docker image

Firstly, put the installer of "Intel Quartus" and "Intel SoCEDS" into "dlk/docker" directory as below.
"Quartus" and "SoCEDS" must be version 17.1 and must NOT rename installers.
Both of them can be downloaded from Intel's website and a registration is necessary for downloading.

    $ cp Quartus-lite-17.1.0.590-linux.tar dlk/docker
    $ cp SoCEDSSetup-17.1.0.590-linux.run dlk/docker

Next, go to the top directory of DLK and run the "docker build" command like below.

    $ cd dlk
    $ docker build -t lm_dlk:0.26 -f docker/Dockerfile .

It will take 90 minutes to make a docker image when 16 cores are used.
It also requires 50GB or more capacity of a disk.
At here, lm_dlk means "REPOSITORY" and 0.26 means "TAG".
You can choose value of "REPOSITORY" and "TAG" freely.
Please refer official document of docker about details.

## Run the docker image

"docker run" is the only necesary operation to start the docker of DLK.
You can execute like below.

    $ docker run --security-opt seccomp=unconfined -v /mnt_host:/mnt_guest -it lm_dlk:0.26

At here, /mnt_host means mount point of host, /mnt_guest means mount point of guest, lm_dlk means "REPOSITORY" and 0.26 means "TAG".
We can pass data between host and guest through the mount point.
Please refer official document of docker about details of -v option and mount point.


# How to use the automated script

The automated script is for performing DLK easily and it's located in "scripts/dlk_script_wrapper.sh" of the DLK.

This document shows how to run DLK with "scripts/dlk_script_wrapper.sh".
We assume that <DLK_DIR> is the top level directory of DLK.


## Prepare input directory and data

Firstly We need to make the directory which includes input data and we assume that the name is <INPUT_DIR> for a explanation.
Of cource you can choose a name as you like. 
It's OK to put <INPUT_DIR> to anywhere you like.
Next we need to prepare "*.pb" file into the <INPUT_DIR>. 
In this time, we named as "example.pb".
Of course other name is OK, but the extention needs to be ".pb".

Now the structure of <INPUT_DIR> becomes as below.

    <INPUT_DIR>
    └── example.pb

<INPUT_DIR> needs to have only one "*.pb" file.

## Prepare output directory

Making output directory is necessary.
We assume that the name of output directory is <OUTPUT_DIR>.
We just need to make <OUTPUT_DIR>, and it needs to be empty to save results.
If the <OUPUT_DIR> directroy is not empty, a error message will be shown.

## Execute the script


We execute the "scripts/dlk_script_wrapper.sh".
"scripts/dlk_script_wrapper.sh" takes three aruguments.

We can execute as below. 

    $ <DLK_DIR>/scripts/dlk_script_wrapper.sh <CLS or DET or SEG> <INPUT_DIR> <OUTPUT_DIR>

If you get an error, please check the error message and fix it.

## Get results

We can get results in the following directory.

    $ ls <OUTPUT_DIR>

You'll see the following files.

* lm_x86.elf
* lm_arm.elf
* lm_fpga.elf
* libdlk_x86.so
* libdlk_arm.so
* libdlk_fpga.so
* <FPGA_DIR>/preloader-mkpimage.bin
* <FPGA_DIR>/soc_system.rbf

<FPGA_DIR> is the directory and this name is changed by first argument of "dlk_script_wrapper.sh" as below.
* CLS ---> <FPGA_DIR> is classification.
* DET ---> <FPGA_DIR> is object_detection.
* SEG ---> <FPGA_DIR> is segmentation.

Please refer to "run_on_board" page about usage of these files.


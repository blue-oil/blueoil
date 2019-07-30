***********
Usage Guide
***********

Main concept of the Blueoil is quite simple. With just 4 steps, you can generate a neural network runtime which runs on FPGA.

1. Prepare training dataset
2. Generate a configuration file
3. Train network according to the config
4. Convert trained result to FPGA ready format

Blueoil is battery included, so you don't need to bother about defining your own network architecture, writing your own data augmentation, etc.

.. toctree::
   :maxdepth: 1

   dataset
   init
   train
   convert
   run
   timing

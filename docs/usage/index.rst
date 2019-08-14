***********
Usage Guide
***********

Main concept of the Blueoil is quite simple: with just 4 steps, you can generate a neural network runtime which can run on an FPGA.

1. Prepare a training dataset.
2. Generate a configuration file.
3. Train a network according to a config.
4. Convert the trained network to an FPGA-ready format.

You could say Blueoil comes "with batteries included", i.e. you don't need to bother about things like  defining your own network architecture, writing your own data augmentation, etc.

.. toctree::
   :maxdepth: 1

   dataset
   init
   train
   convert
   run
   timing

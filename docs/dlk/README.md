# DLK
# Overview
DLK is a tool which automatically convert DNN graph
representation into executable binary working in multi device environments throughout
C source codes including HLS.
Also, it can import lmnet's config and exported test data in order to control
runtime binary behavior automatic test to guarantee the output must return the
same result than that of the input.
<br/>

dlk contains following 3 separable steps.

1. Import lmnet's DNN graph and load graph as DLK graph
2. Optimize DLK graph for the target run-time device
3. Generate codes which will be compiled to the target device
<br/>


# Directory structure
- **python/dlk**: Python scripts here
- **src**: C++ code here
- **docs**: See docs
- **examples**: Test examples include the DNN graph and npy test files
- **tests**: Test scripts automatically run by CI

<br/>

# Current acceptable graph representation
- Tensorflow protocol buffer

<br/>


# Installation


## Setup.py installation
To install DLK run the command:
```
$ python setup.py install
[long output, could take some minutes depending on your environment]
```

If you fail in installing python packages like numpy, tensorflow, etc., try upgrading `setuptools` first, then install DLK again.
```
$ pip install --upgrade setuptools
```


Now you should be able to successfully run:
```
$  PYTHONPATH=python/dlk python python/dlk/scripts/generate_project.py --help
Usage: generate_project.py [OPTIONS]

Options:
...
```

# Getting Started
`generate_project.py` is a script which automatically runs all of above steps
in overview. It imports lmnet's TensorFlow and generate all cpp source headers and other control files like Makefile.
You can quickly carry out this conversion with a classification sample:  `examples/classification/lmnet_quantize_cifar10/minimal_graph_with_shape.pb`.

**CAUTION**
This *Getting Started* is intended only working on CPU not FPGA.
If you need to run the generated binary on FPGA, please follow Custom project generation & Optimizations section.

```
>> PYTHONPATH=python/dlk python python/dlk/scripts/generate_project.py -i examples/classification/lmnet_quantize_cifar10/minimal_graph_with_shape.pb -o tmp/ -p classification
>> cd tmp/classification.prj
>> make lib_x86 -j 4
>> PYTHONPATH=python/dlk python utils/run_test.py  -i <path to input image>/raw_image.png -l <path to generated library>/libdlk_x86.so -e <path to expected output>/xxx_output\:0.npy
```
Here the npy file `xxx_output\:0.npy` starts with a certain number, followed by "_output".

The description of each step in the script are available below, which means you
can run separately each of them.


# Custom project generation & Optimizations
There are several ways to generate a custom project.
Some of them must be applied when you want to run the program on FPGA.
Others are mainly for optimizing the execution like acceleration and memory usage reduction.

### Hard Quantize
This optimization is necessary to execute the graph on FPGA because this
convert the float quantized weight while training, into integer one without breaking any
consistency of the result.
This is available by taking `-hq` flag to `generate_project.py` program which
you can also see in Getting Started section.

#### example
```
>> PYTHONPATH=python/dlk python python/dlk/scripts/generate_project.py -i examples/classification/lmnet_quantize_cifar10/minimal_graph_with_shape.pb -o tmp/ -p classification_hq -hq
```
<!---
### Threshold Skipping
This optimization is special for a graph including quantized operator.
We can skip any operations between convolution and its activation
quantization as long as the operations are monotone function.

#### example
```
PYTHONPATH=python/dlk python python/dlk/scripts/generate_project.py -i examples/classification/lmnet_quantize_cifar10/minimal_graph_with_shape.pb -o tmp/ -p classification_hq_ts -ts -hq
```
-->

# Auto IP synthesis and boot-files generation for FPGA
`blueoil_build_altera.tpl.sh` will also be generated in yor project directory which you
generated from your last command using `generate_project.py`.
Above script is going automatically to run following steps
which are related to run the generated binary on FPGA.

0 - HLS simulation.
1 - HLS synthesis.
2 - Qsys HDL generation
3 - Quartus Compilation
4 - BSP configuration.

The usage is just run the script with the index from which you want to execute.
For example, if you want to run all of these steps, you just need to take 0 as
a 1st argument.
`./blueoil_build_altera.sh 0`

# Test
you can run all tests whose name starts from `test_` under `tests/` dir.
all you need to do is just call
```
PYTHONPATH=python/dlk python setup.py test
```

# Development
This project contains custom `pycodestyle` rule. Please enable it before you start to edit.
I recommend you to activate auto-checking in your editor.

You can easily install some packages to validate your code such as `pycodestyle`.
```
pip3 install -e '.[validation]'
```

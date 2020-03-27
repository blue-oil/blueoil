# Time measurement

When it comes to the `*.elf` test binaries, all of them are built with time measurement enabled, so whenever you run them you should see the times various operations took to complete in the output.

## Preparing `.npy` files

`.npy` files are necessary for running the `*.elf` test binaries. Files with the `*.npy` extension are NumPy Array files. You can check the file format documentation for them [here](https://github.com/numpy/numpy/blob/master/doc/neps/nep-0001-npy-format.rst).

For the running the generated test binaries, two `.npy` files are needed: an example input and the accompanying output. Just grab any example image, run it through your network on your _host_ machine to get your matching output array. Once that is done, all you need to do is use `numpy.save()`.
As an example:
```
import numpy

# Load an imput image from file.
input = load_img("input_image.png")

#Run the network.
output = run_network(input)

# Save the files. Generates 'input.npy' and 'output.npy'.
numpy.save("input", input)
numpy.save("output", output)
```

## Running
Find the file `.elf` you want to run(`lm_fpga.elf`, `lm_arm.elf` or `lm_x86.elf`) in your output directory.

Then, for example,  if you run...

```
./lm_fpga.elf input.npy output.npy
```

...on an FPGA, you should see output like...

```
-------------------------------------------------------------
Comparison: Default network test  succeeded!!!
-------------------------------------------------------------
TotalInitTime 1.18751e+06,  sum:1187.51ms
TotalRunTime 504207,  sum:504.207ms
..ExtractImagePatches 27700,2007,579,  sum:30.286ms
..Convolution 78843,  sum:78.843ms
....kn2row-1x1 78824,  sum:78.824ms
......matrix_multiplication 78803,  sum:78.803ms
..BatchNorm 5667,  sum:5.667ms
..LinearMidTreadHalfQuantizer 36668,  sum:36.668ms
....pack_input 26204,  sum:26.204ms
..QuantizedConv2D 8048,5308,17001,17071,17069,17052,9487,19927,25026,24564,  sum:160.553ms
....Tensor convert 2995,319,290,380,373,367,307,862,4145,3727,  sum:13.765ms
....Sync UDMABuf Input 598,577,587,581,578,581,583,978,2764,2770,  sum:10.597ms
....Conv2D TCA 4094,4045,15763,15748,15747,15742,7957,15740,15714,15710,  sum:126.26ms
....Sync UDMABuf Output 322,338,332,336,338,336,614,2316,2360,2326,  sum:9.618ms
..Memcpy 448,448,569,457,522,448,932,3761,3736,  sum:11.321ms
..DepthToSpace 3147,6952,28072,  sum:38.171ms
..linear_to_float 120701,  sum:120.701ms
```

## Parsing the output
Each line start with the operation in question, followed by a list of times in nanoseconds for each run of that particular operations. At the end of each line we have the sum milliseconds of the time spent on that operation.

The total runtime is indicated by `TotalRunTime`, while the time spent on initialization is shown under `TotalInitTime`.

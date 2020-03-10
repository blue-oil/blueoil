# Shared libraries

## Generation
After you generate your project from your protocol buffer, you can build libraries for each platform that the program is going to work on. 
The compile command would be like below.

```
make lib_x86
make lib_arm
make lib_fpga
```

These commands will generate the following files:
```
libdlk_x86.so
libdlk_arm.so
libdlk_fpga.so
```

After generating the shared librariues, you can use them from, for example, Python and C++.

## Usage
You can use the generated library from python with the helper script provided in `blueoil/converter/nnlib`.

```python
# utils/run_inference.py
import numpy as np
from sys import argv
from os import path
from PIL import Image
from nnlib import NNLib


input_image_path = argv[1]
lib_path = argv[2]


if __name__ == '__main__':
    # load and initialize the generated shared library
    nn = NNLib()
    nn.load(lib_path)
    nn.init()

    # load an image
    img = Image.open(input_image_path).convert('RGB')
    img.load()

    data = np.asarray(img, dtype=np.float32)
    data = np.expand_dims(data, axis=0)

    # apply the preprocessing, which is DivideBy255 in this case
    data = data / 255.0

    # run the graph and show output
    output = nn.run(data)
    print(f'Output: {output}')
```

In above sample code `utils/run_inference.py`, you can run the graph with an input image and generated library that you take to the script.
The result would be like below.

```
> python utils/run_inference.py input_image.png ./libdlk_x86.so
Output: [[0.10000502 0.10004324 0.09995745 0.10000631 0.10003947 0.09993229 0.10000196 0.09998367 0.1000008  0.10002969]]
```


## Another sample code
There is also a good sample in `utils/run_test.py`.
There you can also compare the result with the data you expect.

For example, there is `expected.npy` that is the output tensor data you expect from the graph we described in Usage section.
```
> python -c 'import numpy; print(numpy.load("expected.npy"))'
[[0.10000502 0.10004324 0.09995745 0.10000631 0.10003947 0.09993229 0.10000196 0.09998367 0.1000008  0.10002969]]
```

When you run it with the input image and the library, the result would be like below.

```
> python utils/run_test.py -i input_image.png -l ./libdlk_x86.so -e expected.npy
Output:
[[0.10000502 0.10004324 0.09995745 0.10000631 0.10003947 0.09993229 0.10000196 0.09998367 0.1000008 0.10002969]]
Test: 100.000% of the output values are correct
```

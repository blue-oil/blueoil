# Archive file

## Generation
After you generate your project from your protocol buffer, you can pack binaries into an archive for each platform that the program is going to work on. 
The compile command would be like below.

```
make build ARCH={x86, x86_avx, aarch64, arm} USE_FPGA={enable, disable} TYPE=static
```

USE_FPGA is valid only for ARCH={aarch64, arm}.
If there's no USE_FPGA option, it's equal to USE_FPGA=disable option.

Example:
```
make clean
make build ARCH=x86 TYPE=static
```

These commands will generate the following files:

* libdlk_x86.a
* libdlk_x86_avx.a
* libdlk_aarch64.a
* libdlk_aarch64_fpga.a
* libdlk_arm.a
* libdlk_arm_fpga.a

After you generate one of above arhive files, you can compile it with your own source codes.
The defined functions are same than those of the shared libraries (.so).

## Usage
Here we show the example that generates binary behaves same than `lm_x86` for classificatoin.


#### Generate Project (common process)
```
python blueoil/converter/generate_project.py -i examples/classification/lmnet_quantize_cifar10/minimal_graph_with_shape.pb -o ./ -p cls -hq
cd cls.prj
```

#### Compile for x86
```
make clean
make build ARCH=x86 TYPE=static -j8
g++ -std=c++14 mains/main.cpp libdlk_x86.a -I./include -o lm_x86.elf
```

#### Compile for x86_avx
```
make clean
make build ARCH=x86_avx TYPE=static -j8
g++ -std=c++14 mains/main.cpp libdlk_x86_avx.a -I./include -o lm_x86_avx.elf -fopenmp
```

#### Compile for aarch64
```
make clean
make build ARCH=aarch64 TYPE=static -j8
aarch64-linux-gnu-g++ -std=c++14 mains/main.cpp libdlk_aarch64.a -I./include -o lm_aarch64.elf -fopenmp
```

#### Compile for aarch64_fpga
```
make clean
make build ARCH=aarch64 USE_FPGA=enable TYPE=static -j8
aarch64-linux-gnu-g++ -std=c++14 mains/main.cpp libdlk_aarch64_fpga.a -I./include -o lm_aarch64_fpga.elf -fopenmp
```

#### Compile for arm
```
make clean
make build ARCH=arm TYPE=static -j8 
arm-linux-gnueabihf-g++ -std=c++14 mains/main.cpp libdlk_arm.a -I./include -o lm_arm.elf -fopenmp
```

#### Compile for arm_fpga
```
make clean
make build ARCH=arm USE_FPGA=enable TYPE=static -j8
arm-linux-gnueabihf-g++ -std=c++14 mains/main.cpp libdlk_arm_fpga.a -I./include -o lm_arm_fpga.elf -fopenmp
```

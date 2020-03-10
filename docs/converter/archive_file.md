# Archive file

## Generation
After you generate your project from your protocol buffer, you can pack binaries into an archive for each platform that the program is going to work on. 
The compile command would be like below.

```
make ar_x86
make ar_arm
make ar_fpga
```

These commands will generate the following files:
```
libdlk_x86.a
libdlk_arm.a
libdlk_fpga.a
```

After you generate one of above arhive files, you can compile it with your own source codes.
The defined functions are same than those of the shared libraries (.so).

## Usage
Here we show the example that generates binary behaves same than `lm_x86` for classificatoin.


#### generate project (common process)
```
>> python blueoil/converter/generate_project.py -i examples/classification/lmnet_quantize_cifar10/minimal_graph_with_shape.pb -o ./ -p cls -hq
>> cd cls.prj
```

#### for x86
```
>> make ar_x86 -j8 # this generates libdlk_x86.a
>> g++ -std=c++11 mains/main.cpp libdlk_x86.a -I./include -o lm_x86_from_ar.elf -pthread -fopenmp
>> ./lm_x86_from_ar.elf <debug input .npy file> <debug output .npy file>
-------------------------------------------------------------
comparison: default network test  succeeded!!!
-------------------------------------------------------------
```

#### for arm
```
>> make ar_arm -j8 # this generates libdlk_arm.a
>> arm-linux-gnueabihf-g++
 -std=c++11 mains/main.cpp libdlk_arm.a -I./include -o lm_arm_from_ar.elf -pthread -fopenmp
>> scp lm_arm_from_ar.elf ${your DE10-Nano board}
>> ssh ${your DE10-Nano board}
>> ./lm_arm_from_ar.elf <debug input .npy file> <debug output .npy file>
```

#### for fpga
```
>> make ar_fpga -j8 # this generates libdlk_arm.a
>> arm-linux-gnueabihf-g++
 -std=c++11 mains/main.cpp libdlk_fpga.a -I./include -o lm_fpga_from_ar.elf -pthread -fopenmp
>> scp ./lm_fpga_from_ar.elf ${your DE10-Nano board}
>> ssh ${your DE10-Nano board}
>> ./lm_fpga_from_ar.elf <debug input .npy file> <debug output .npy file>
```

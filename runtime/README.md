# Import model with dlk library

- Choose a model to try from blueoil/dlk/examples/
   - for example: classification/lmnet_quantize_cifar10
- make libdlk_x86.a with reference blueoil/dlk/README.md
- put libdlk_x86.a & meta.yaml to blueoil/runtime/examples/

# Build blueoil static lib
```
$ mkdir build
$ cd build
# You can set `DLK_LIB_DIR` environment.
$ DLK_LIB_DIR=`pwd`/../examples/ cmake ../
$ make
$ make install
$ tree output/
output/
├── include
│   └── blueoil.hpp
└── lib
    └── libblueoil.a
```


# Build blueoil shared lib
```
$ mkdir build
$ cd build
# -DBUILD_SHARED_LIBS=ON
$ DLK_LIB_DIR=`pwd`/../examples/ cmake -DBUILD_SHARED_LIBS=ON ../
$ make
$ make install
$ tree output/
output/
├── include
│   └── blueoil.hpp
└── lib
    ├── libblueoil.so -> libblueoil.so.0.1.0
    └── libblueoil.so.0.1.0
```


# How to run example.

```
$ cd examples
# copy builded static lib and header.
$ cp -R ../build/output/* ./
$ cmake .
$ make
$ ./run -i cat.npy -c meta.yaml
classes:
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
task: IMAGE.CLASSIFICATION
expected input shape:
1
32
32
3
Run
Results !
shape:1 10
0.000105945 6.23502e-05 0.0323531 0.00360625 0.0124029 0.000231775 0.951004 8.7062e-05 9.84179e-05 4.80589e-05
```

# Unit tests

```
$ cd build
$ DLK_LIB_DIR=`pwd`/../examples/ cmake -DBUILD_SHARED_LIBS=ON ../
$ make
$ make test
Running tests...
Test project <repos_dir>/blueoil/runtime/build
    Start 1: blueoil-test-tensor
1/5 Test #1: blueoil-test-tensor ..............   Passed    0.00 sec
    Start 2: blueoil-test-image
2/5 Test #2: blueoil-test-image ...............   Passed    0.00 sec
    Start 3: blueoil-test-npy
3/5 Test #3: blueoil-test-npy .................   Passed    0.00 sec
    Start 4: blueoil-test-resize
4/5 Test #4: blueoil-test-resize ..............   Passed    0.00 sec
    Start 5: blueoil-test-data_processor
5/5 Test #5: blueoil-test-data_processor ......   Passed    0.00 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   0.08 sec
```

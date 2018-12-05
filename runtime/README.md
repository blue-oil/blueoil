# Build blueoil static lib
```
$ mkdir build
$ cd build
# You can set `DLK_LIB_DIR` environment.
$ DLK_LIB_DIR=`pwd`/../examples/dlk_lib/ cmake ../
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
$ DLK_LIB_DIR=`pwd`/../examples/dlk_lib/ cmake -DBUILD_SHARED_LIBS=ON ../
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
$ cmake
$ make
$ ./a.out
classes:
0
1
2
3
4
5
6
7
8
9
task: IMAGE.CLASSIFICATION
expected input shape:
1
128
128
3
Run
Results !
0.100382
0.0998879
0.101126
0.098727
0.100418
0.0999296
0.100612
0.0995238
0.0996848
0.0997096
```

Currentlly pre/post-process functions are NOT correctly implemented. 
The value of Results! 0.100382, 0.0998879 will be changed after correct implementation.


# Unit tests

```
$ cd build
$ DLK_LIB_DIR=`pwd`/../examples/dlk_lib/ cmake -DBUILD_SHARED_LIBS=ON ../
$ make
$ make test
Running tests...
Test project /home/yoya/git/yoya/blueoil/runtime/build
    Start 1: blueoil-test-tensor
1/4 Test #1: blueoil-test-tensor ..............   Passed    0.00 sec
    Start 2: blueoil-test-image
2/4 Test #2: blueoil-test-image ...............   Passed    0.00 sec
    Start 3: blueoil-test-opencv
3/4 Test #3: blueoil-test-opencv ..............   Passed    0.04 sec
    Start 4: blueoil-test-resize
4/4 Test #4: blueoil-test-resize ..............   Passed    0.04 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =   0.08 sec
```

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

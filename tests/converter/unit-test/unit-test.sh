#!/bin/bash
cd tests/converter/unit-test
mkdir build
cd build
cmake ..
make
./converter_unit_test
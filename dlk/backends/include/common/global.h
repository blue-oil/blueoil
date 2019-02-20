/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include "common/type_info.h"

#include "params/conv.h"

#define IP_CSR_ADDR 0xFF200000
#define A8W1_IP_CSR_ADDR 0xFF200100
#define IN_DATA_ADDR 0x20000000
#define OUT0_DATA_ADDR 0x2C000000
#define OUT1_DATA_ADDR 0x32000000
#define OUT_DATA_ADDR OUT0_DATA_ADDR
#define K_DATA_ADDR 0x38000000
#define THRESHOLDS_ADDR 0x3F000000

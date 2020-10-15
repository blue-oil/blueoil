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
=============================================================================*/

#include "network.h"


extern "C" __attribute__ ((visibility ("default"))) Network* network_create()
{
  return new Network();
}

extern "C" __attribute__ ((visibility ("default"))) void network_delete(Network *nn)
{
  if(nn != nullptr)
    delete nn;
}

extern "C" __attribute__ ((visibility ("default"))) bool network_init(Network *nn)
{
  return nn->init();
}

extern "C" __attribute__ ((visibility ("default"))) int network_get_input_rank(Network *nn)
{
  return nn->get_input_rank();
}

extern "C" __attribute__ ((visibility ("default"))) int network_get_output_rank(Network *nn)
{
  return nn->get_output_rank();
}

extern "C" __attribute__ ((visibility ("default"))) void network_get_input_shape(Network *nn, int32_t *shape)
{
  nn->get_input_shape(shape);
}

extern "C" __attribute__ ((visibility ("default"))) void network_get_output_shape(Network *nn, int32_t *shape)
{
  nn->get_output_shape(shape);
}

extern "C" __attribute__ ((visibility ("default"))) void network_run(Network *nn, void *input, void *output)
{
  nn->run(input, output);
}

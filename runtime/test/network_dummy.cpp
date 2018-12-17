#include "blueoil.hpp"

extern "C" { // dummy functions
  Network *network_create() { return NULL; }
  bool network_init(Network *) { return true; }
  int network_get_input_rank(const Network *) { return 0; }
  int network_get_output_rank(const Network *) { return 0; }
  void network_get_input_shape(const Network *, int *) { ; }
  void network_get_output_shape(const Network *, int *) { ; }
  void network_run(Network *, const float *, float *) { ; }
}

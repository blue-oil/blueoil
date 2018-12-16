#include "blueoil.hpp"

extern "C" { // dummy functions
  Network *network_create() { return NULL; }
  int network_get_input_rank(Network *) { return 0; }
  void network_get_input_shape(Network *, int *) { ; }
  int network_get_output_rank(Network *) { return 0; }
  void network_get_output_shape(Network *, int *) { ; }
  bool network_init(Network *) { return true; }
  void network_run(Network *, float *, float *) { ; }
}

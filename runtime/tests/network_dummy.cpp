#include "blueoil.hpp"

extern "C" { // dummy functions
    Network *network_create() { return NULL; }
    int network_get_input_rank(Network *nn) { (void) nn ; return 0; }
    void network_get_input_shape(Network *nn, int *shape) { (void)nn ; (void)shape; }
    int network_get_output_rank(Network *nn) { (void)nn ; return 0; }
    void network_get_output_shape(Network *nn, int *shape) { (void)nn; (void)shape; }
    bool network_init(Network *nn) { (void) nn ; return true; }
    void network_run(Network *nn, float *input, float *output) { (void)nn; (void)input; (void)output; }
}

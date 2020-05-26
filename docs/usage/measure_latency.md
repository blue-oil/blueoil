# Measuring latency
Measure the average latency of certain model's prediction at runtime.

The latency is averaged over number of repeated executions -- by default is to run it 100 times.
Each execution is measured after tensorflow is already initialized and both model and images are loaded.
Batch size is always 1.

Measure two types latency,
First is `overall` (including pre-post-processing which is being executed on CPU), Second is `network-only` (model inference, excluding pre-post-processing).

```
Options:
  -c, --config_file TEXT          config file path.
                                  When experiment_id is
                                  provided, The config override saved
                                  experiment config. When experiment_id is
                                  provided and the config is not provided,
                                  restore from saved experiment config.
  -i, --experiment_id TEXT        id of this experiment.
  --restore_path TEXT             restore ckpt file base path. e.g. saved/expe
                                  riment/checkpoints/save.ckpt-10001.
  --image_size <INTEGER INTEGER>...
                                  input image size height and width. if it is
                                  not provided, it restore from saved
                                  experiment config.
  -n, --step_size INTEGER         number of execution (number of samples).
                                  default is 100.
  --cpu                           flag use only cpu
  -h, --help                      Show this message and exit.
```

e.g.
```
$ python blueoil/cmd/measure_latency.py -c configs/core/object_detection/lm_fyolo_quantize_pascalvoc_2007_2012.py -n 20 --image_size 160 160

...
---- measure latency result ----
total number of execution (number of samples): 20
network: LMFYoloQuantize
use gpu by network: True
image size: [160, 160]
devices: ['device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0, compute capability: 5.2']

* overall (include pre-post-process which execute on cpu)
total time: 101.7051 msec
latency
   mean (SD=standard deviation): 5.0853 (SD=0.1654) msec, min: 4.9169 msec, max: 5.5995 msec
FPS
   mean (SD=standard deviation): 196.8454 (SD=6.1032), min: 178.5874, max: 203.3799

* network only (exclude pre-post-process):
total time: 78.3966 msec
latency
   mean (SD=standard deviation): 3.9198 (SD=0.1583) msec, min: 3.7389 msec, max: 4.4239 msec
FPS
   mean (SD=standard deviation): 255.5047 (SD=9.7093), min: 226.0471, max: 267.4598
---- measure latency result ----
```

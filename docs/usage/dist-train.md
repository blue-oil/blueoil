# Train a neural network with multi GPUs


```
# Select GPUs
$ export CUDA_VISIBLE_DEVICES=0,1,2,3
# Build docker image for distributed training
$ ./docker_build.sh -dist
# Run distributed training
$ ./blueoil.sh dist-train config/test.yml 4

Usage:
  ./blueoil.sh dist-train <YML_CONFIG_FILE> <NUM_WORKERS> <OUTPUT_DIRECTORY(optional)> <EXPERIMENT_ID(optional)>

Arguments:
   YML_CONFIG_FILE       config file path for this training  [required]
   NUM_WORKERS           the number of workers for distribution
   OUTPUT_DIRECTORY      output directory path for saving models [optional] (defalt is ./saved)
   EXPERIMENT_ID         id of this training [optional] (default is {CONFIG_NAME}_{TIMESTAMP})
```

`blueoil dist-train` command runs actual training with multi GPUs.

Before running `blueoil dist-train` command, make sure you've already put training/test data in the proper path, defined in configuration file.

If you want to stop training, you should type `Ctrl + C` or kill `blueoil.sh` processes. And you can restart training from saved checkpoints with setting `EXPERIMENT_ID` to be the same as existing id.

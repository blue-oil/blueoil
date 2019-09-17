# Train a neural network


```
$ ./blueoil.sh train config/test.yml

Usage:
  ./blueoil.sh train <YML_CONFIG_FILE> <OUTPUT_DIRECTORY(optional)> <EXPERIMENT_ID(optional)>

Arguments:
   YML_CONFIG_FILE       config file path for this training  [required]
   OUTPUT_DIRECTORY      output directory path for saving models [optional] (defalt is ./saved)
   EXPERIMENT_ID         id of this training [optional] (default is {CONFIG_NAME}_{TIMESTAMP})
```

`blueoil train` command runs actual training.

Before running `blueoil train`, make sure you've already put training/test data in the proper location, as defined in the configuration file.

If you want to stop training, you should press `Ctrl + C` or kill the `blueoil.sh` processes. You can restart training from saved checkpoints by setting `EXPERIMENT_ID` to be the same as an existing id.

## Training on GPUs

Bueoil supports tranining on CUDA enabled GPUs. To train on a GPU, notify a GPU ID to use by the environment variable `CUDA_VISIBLE_DEVICES`. For example, if you want to use GPU ID 0, you should set the environment variable as `CUDA_VISIBLE_DEVICES="0"`.

Blueoil also support multiple GPU training using Horovod. For example, if you want to use GPU ID 0 and GPU ID 1, then you should set the environment variable as `CUDA_VISIBLE_DEVICES="0,1"`. Internally, Blueoil count the number of "," from `CUDA_VISIBLE_DEVICES`, to count the number of GPUs. (Hence, `CUDA_VISIBLE_DEVICES='0,,1,,2,,3'` would confuse Blueoil.) If the counted number of GPUs are greater than 1, then Blueoil automatically prepends `horovodrun` command to enable multiple GPU training.

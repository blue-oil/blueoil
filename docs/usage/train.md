# Train a neural network


```
$ PYTHONPATH=.:blueoil:blueoil/cmd python blueoil/cmd/main.py train -c config/test.py

Usage:
  main.py train [OPTIONS]

Arguments:
  -c, --config TEXT         Path of config file.  [required]
  -e, --experiment_id TEXT  ID of this training.
  --help                    Show this message and exit.
```

`python blueoil/cmd/main.py train` command runs actual training.

Before running `python blueoil/cmd/main.py train`, make sure you've already put training/test data in the proper location, as defined in the configuration file.

If you want to stop training, you should press `Ctrl + C` or kill the `blueoil train` processes. You can restart training from saved checkpoints by setting `experiment_id` to be the same as an existing id.

## Training on GPUs

Bueoil supports tranining on CUDA enabled GPUs. To train on a GPU, notify a GPU ID to use by the environment variable `CUDA_VISIBLE_DEVICES`. For example, if you want to use GPU ID 0, you should set the environment variable as `CUDA_VISIBLE_DEVICES="0"`.

Blueoil also support multiple GPU training using Horovod. For example, if you want to use GPU ID 0 and GPU ID 1, then you should set the environment variable as `CUDA_VISIBLE_DEVICES="0,1"`. Internally, Blueoil count the number of "," from `CUDA_VISIBLE_DEVICES`, to count the number of GPUs. (Hence, `CUDA_VISIBLE_DEVICES='0,,1,,2,,3'` would confuse Blueoil.) If the counted number of GPUs are greater than 1, then Blueoil automatically prepends `horovodrun` command to enable multiple GPU training.

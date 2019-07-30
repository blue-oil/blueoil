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

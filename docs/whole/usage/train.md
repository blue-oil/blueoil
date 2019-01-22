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

Before running `blueoil train` command, make sure you've already put training/test data in the proper path, defined in configuration file.

If you want to stop training, you should type `Ctrl + C` or kill `blueoil.sh` processes. And you can restart training from saved checkpoints with setting `EXPERIMENT_ID` to be the same as existing id.

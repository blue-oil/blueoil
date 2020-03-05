# Profiling model
Profiling a trained model.

If it exists unquantized layers, use `-uql` to point it out.

```
# python blueoil/cmd/profile_model.py -h
Usage: profile_model.py [OPTIONS]

  Profiling a trained model.

  If it exists unquantized layers, use `-uql` to point it out.

Options:
  -i, --experiment_id TEXT     id of this experiment.
  --restore_path TEXT          restore ckpt file base path. e.g.
                               saved/experiment/checkpoints/save.ckpt-10001
  -c, --config_file TEXT       config file path.
                               When experiment_id is
                               provided, The config override saved experiment
                               config. When experiment_id is provided and the
                               config is not provided, restore from saved
                               experiment config.
  -b, --bit INTEGER            quantized bit
  -uql, --unquant_layers TEXT  unquantized layers
  -h, --help                   Show this message and exit.
```

e.g.
`python blueoil/cmd/profile_model.py -i lmnet_cifar10 --restore_path saved/lmnet_cifar10/checkpoints/save.ckpt-99001 --bit 2 -uql conv1 -uql conv7`

`python blueoil/cmd/executor/profile_model.py -c configs/core/classification/lmnet_cifar10.py --bit 1`

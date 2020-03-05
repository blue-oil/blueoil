# Evaluating model
Main evaluation script is `blueoil/cmd/evaluate.py`.
When you start doing evaluation, you need to specify some (required) options.
You can see the option descriptions with `-h` flag.
```
# python blueoil/cmd/evaluate.py -h
Usage: evaluate.py [OPTIONS]

Options:
  -i, --experiment_id TEXT  id of this training  [required]
  --restore_path TEXT       restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001
  -n, --network TEXT        network name. override config.NETWORK_CLASS
  -d, --dataset TEXT        dataset name. override config.DATASET_CLASS
  -c, --config_file TEXT    config file path. override(merge) saved experiment config.
                            if it is not provided, it restore from saved experiment config.
  -o, --output_dir TEXT     Output directory to save a evaluated result
  -h, --help                Show this message and exit.
```

`--network` and `--dataset` option override config on the fly.

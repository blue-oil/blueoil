# Exporting model to proto buffer
Exporting a trained model to proto buffer files and meta config yaml.

In the case with `images` option, create each layer output value npy files in `export/{restore_path}/{image_size}/{image_name}/**.npy` for debug.

* Load config file from saved experiment dir.
* Export config file to yaml. See also [Config specification](../specification/config.md).
  * `config.yaml` can be used for training and evaluation in python. i.e. [classification.yaml](../../blueoil/configs/example/classification.yaml) is exported from [classification.py](../../blueoil/configs/example/classification.py)
  * `meta.yaml` include only few parameter for application such as demo. i.e. [classification_meta.yaml](../../blueoil/configs/example/classification_meta.yaml) is exported from [classification.py](../../blueoil/configs/example/classification.py)
* Save the model protocol buffer files (tf) for DLK converter.
* Output each layer npy files for DLK converter debug.
* Write summary in tensorboard `export` dir.

```
# python blueoil/cmd/export.py -h
Usage: export.py [OPTIONS]

  Exporting a trained model to proto buffer files and meta config yaml.

  In the case with `images` option, create each layer output value npy files
  in `export/{restore_path}/{image_size}/{image_name}/**.npy` for debug.

Options:
  -i, --experiment_id TEXT        id of this experiment.  [required]
  --restore_path TEXT             restore ckpt file base path. e.g.
                                  saved/experiment/checkpoints/save.ckpt-10001
  --image_size <INTEGER INTEGER>...
                                  input image size height and width. if it is
                                  not provided, it restore from saved
                                  experiment config. e.g. --image_size 320 320
  --images TEXT                   path of target images
  -c, --config_file TEXT          config file path. override saved experiment
                                  config.
  -h, --help                      Show this message and exit.
```

e.g.
`python blueoil/cmd/export.py -i lmnet_cifar10 --restore_path saved/lmnet_cifar10/checkpoints/save.ckpt-99001 --images apple_128.png --images apple.png --image_size 128 192`

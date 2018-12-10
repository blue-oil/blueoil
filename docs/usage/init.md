# Generate a configuration file

You can generate your configuration file interactively by running `blueoil init` command.

    $ ./blueoil.sh init

`blueoil init` command generates a configuration file to train your new model.

With the `blueoil init` command, you can configure your training configuration interactively.

This is an example of configuration.
```
#### Generate config ####
  your model name ():  test
  choose task type  classification
  choose network  LmnetV1Quantize
  choose dataset format  Caltech101
  training dataset path:  {dataset_dir}/train/
  set validataion dataset? (if answer no, the dataset will be separated for training and validation by 9:1 ratio.)  yes
  validataion dataset path:  {dataset_dir}/test/
  batch size (integer):  64
  image size (integer x integer):  32x32
  how many epochs do you run training (integer):  100
  select optimizer: MomentumOptimizer
  initial learning rate: 0.001
  message': 'choose learning rate setting(tune1 / tune2 / tune3 / fixed): tune1 -> "2 times decay"
  apply quantization at the first layer: yes

#### how can I change small setting? Or I need to re-run `blueoil init` again?

`blueoil init` just generates a config file in YAML format. You can change some settings, according to comments.


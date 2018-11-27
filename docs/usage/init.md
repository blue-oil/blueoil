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
  enable data augmentation?  No
  training dataset path:  {dataset_dir}/train/
  set validataion dataset? (if answer no, the dataset will be separated for training and validation by 9:1 ratio.)  yes
  validataion dataset path:  {dataset_dir}/test/
  batch size (integer):  64
  image size (integer x integer):  32x32
  how many epochs do you run training (integer):  100
  apply quantization at the first layer: yes
```

#### how can I change small setting? Or I need to re-run `blueoil init` again?
You don't need to re-run `bluoil init` again.
`blueoil init` just generates a config file in YAML format. You can change some settings, according to comments.

#### How to get details about data augmentation?

Please see <a href="../reference/data_augmentor.html">here</a> for reference.

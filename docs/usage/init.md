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
  initial learning rate: 0.001
  message': 'choose learning rate setting(tune1 / tune2 / tune3 / fixed): tune1 -> "2 times decay"
  enable data augmentation?  No
  apply quantization at the first layer: yes
```

#### how can I change small setting? Or I need to re-run `blueoil init` again?
You don't need to re-run `bluoil init` again.
`blueoil init` just generates a config file in YAML format. You can change some settings, according to comments.

#### data augmentation
You can use various augmentation methods in generated YAML, also you can change augmentation methods's parameter.
Under `commmon.data_augmentation` in generated yaml, augmentation methods are listed and the parameters are nested in each methods.


generated yaml:
```
common:
  data_augmentation:
    - Blur:
        - value: (0, 1)
    - Color:
        - value: (0.75, 1.25)
    - Contrast:
        - value: (0.75, 1.25)
    - FlipLeftRight:
        - probability: 0.5
```

Please see <a href="../reference/data_augmentor.html">the data augmentor reference page</a>, when you want to know about augmentation methods. the all of augmentation methods and parameter are explained, methods name in generated yaml correspond to class name under `lmnet.data_augmentor` in the reference.

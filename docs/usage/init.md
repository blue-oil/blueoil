# Generate a configuration file

You can generate your configuration file interactively by running `python blueoil/cmd/main.py init`.

    $ PYTHONPATH=.:blueoil:blueoil/cmd python blueoil/cmd/main.py init

`python blueoil/cmd/main.py init` generates a configuration file used to train your new model.

Below is an example.
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
  choose learning rate schedule ({epochs} is the number of training epochs you entered before):  '2-step-decay' -> learning rate decrease by 1/10 on {epochs}/2 and {epochs}-1.
  enable data augmentation?  No
  apply quantization at the first layer: yes

#### how can I change small setting? Or I need to re-run `blueoil init` again?
You don't need to re-run `bluoil init` again.
`blueoil init` just generates a config file in YAML format. You can change some settings, according to comments.

#### data augmentation
You can use various augmentation methods in generated YAML, also you can change augmentation methods's parameter.
Under `commmon.data_augmentation` in generated yaml, augmentation methods are listed and the parameters are nested in each methods.
```

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

Please see <a href="../reference/data_augmentor.html">the data augmentor reference page</a>, when you want to know about augmentation methods. All of the augmentation methods and the required parameters are explained there. Method names in the generated yaml file correspond to class names under `blueoil.data_augmentor` in the reference.


#### optimizer

You can choose an optimizer between Adam and Momentum. Each optimizer uses the TensorFlow implementation. Please see the TensorFlow documentation for [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) and [MomentumOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer).

The generated yaml:

```
trainer:
  ...
  # supported 'optimizer' is 'Momentum', 'Adam' currently.
  # Momentum
  #    https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer
  # Adam
  #    https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
  optimizer: Adam
```

Blueoil uses learning rate you input for both Adam and Momentum. Blueoil uses `momentum=0.9` for Momentum. Other values are TensorFlow defaults as described in the TensorFlow documentation.

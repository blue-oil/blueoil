import tensorflow as tf
import keras
import keras.layers as KL
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

import functools
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

#####################
# QUANTIZE
#####################

ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
WEIGHT_QUANTIZER_KWARGS = {}


#####################
# QUANTIZE
#####################

def quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
    """Get the quantized variables.
    Use if to choose or skip the target should be quantized.
    Args:
        getter: Default from tensorflow.
        name: Default from tensorflow.
        weight_quantization: Callable object which quantize variable.
        args: Args.
        kwargs: Kwargs.
    """
    assert callable(weight_quantization)
    var = getter(name, *args, **kwargs)
    with tf.compat.v1.variable_scope(name):
        # Apply weight quantize to variable whose last word of name is "kernel".
        if "kernel" == var.op.name.split("/")[-1]:
            return weight_quantization(var)
    return var


my_activation = ACTIVATION_QUANTIZER(**ACTIVATION_QUANTIZER_KWARGS)
my_activation = KL.Lambda(my_activation)
weight_quantization = WEIGHT_QUANTIZER(**WEIGHT_QUANTIZER_KWARGS)
my_custom_getter = functools.partial(quantized_variable_getter,
                                     weight_quantization=weight_quantization)


############################################################
#  Resnet Graph
############################################################

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def identity_block_18(input_tensor, kernel_size, filters, stage, block,
                      use_bias=True, train_bn=True, config=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.compat.v1.variable_scope('quantize_group_' + str(stage) + block, custom_getter=my_custom_getter):
        x = KL.Conv2D(nb_filter1, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
        x = my_activation(x)

        x = KL.Conv2D(nb_filter2, (1, 1), name=conv_name_base + '2b',
                      use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = my_activation(x)
    return x


def conv_block_18(input_tensor, kernel_size, filters, stage, block,
                  strides=(2, 2), use_bias=True, train_bn=True, config=None):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.compat.v1.variable_scope('quantize_group_' + str(stage) + block, custom_getter=my_custom_getter):
        x = KL.MaxPooling2D((1, 1), strides=strides)(input_tensor)
        x = KL.Conv2D(nb_filter1, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2a', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
        x = my_activation(x)

        x = KL.Conv2D(nb_filter2, (1, 1), name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)

        shortcut = KL.Conv2D(nb_filter2, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

        x = KL.Add()([x, shortcut])
        x = my_activation(x)
    return x


def resnet_graph_18(input_image, train_bn=True, config=None):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = my_activation(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block_18(x, 3, [64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn, config=config)
    C2 = x = identity_block_18(x, 3, [64, 256], stage=2, block='b', train_bn=train_bn, config=config)
    # Stage 3
    x = conv_block_18(x, 3, [128, 512], stage=3, block='a', train_bn=train_bn, config=config)
    C3 = x = identity_block_18(x, 3, [128, 512], stage=3, block='b', train_bn=train_bn, config=config)
    # Stage 4
    x = conv_block_18(x, 3, [256, 1024], stage=4, block='a', train_bn=train_bn, config=config)
    C4 = x = identity_block_18(x, 3, [256, 1024], stage=4, block='b', train_bn=train_bn, config=config)
    # Stage 5
    x = conv_block_18(x, 3, [512, 2048], stage=5, block='a', train_bn=train_bn, config=config)
    C5 = x = identity_block_18(x, 3, [512, 2048], stage=5, block='b', train_bn=train_bn, config=config)
    return x, C1, C2, C3, C4, C5


input_image = KL.Input(shape=[None, None, 3], name="input_image")

x, _, _, _, _, _ = resnet_graph_18(input_image, train_bn=True, config=None)

# x = KL.AveragePooling2D((7, 7), name='avg_pool')(x)
x = KL.GlobalAveragePooling2D(name='global_avg_pool')(x)
outputs = KL.Dense(1000, activation='softmax', name='fc1000')(x)

model = keras.Model(input_image, outputs, name='resnet18')

log_dir = '/home/zhang/blueoil/lmnet/lmnet/networks/instance_segmentation/logs/'
data_dir = '/storage/dataset/ILSVRC2012/'

BATCH_SIZE = 128
mean = [0.485, 0.456, 0.406]  # rgb
std = [0.229, 0.224, 0.225]

train_gen = ImageDataGenerator(rescale=1 / 255.,
                               width_shift_range=0.125,
                               height_shift_range=0.125,
                               fill_mode='constant',
                               cval=0.,
                               horizontal_flip=True,
                               dtype='float32',
                               preprocessing_function=lambda image: (image - mean) / std)
train_data = train_gen.flow_from_directory(data_dir + 'train', target_size=(256, 256), class_mode='categorical',
                                           batch_size=BATCH_SIZE)


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


train_data = crop_generator(train_data, 224)

from keras.utils import Sequence

# class DataGenerator(Sequence):
#     """Generates data for Keras
#     Sequence based data generator. Suitable for building data generator for training and prediction.
#     """
#     def __init__(self, list_IDs, labels, image_path, mask_path,
#                  to_fit=True, batch_size=32, dim=(256, 256),
#                  n_channels=1, n_classes=10, shuffle=True):
#         """Initialization
#         :param list_IDs: list of all 'label' ids to use in the generator
#         :param labels: list of image labels (file names)
#         :param image_path: path to images location
#         :param mask_path: path to masks location
#         :param to_fit: True to return X and y, False to return X only
#         :param batch_size: batch size at each iteration
#         :param dim: tuple indicating image dimension
#         :param n_channels: number of image channels
#         :param n_classes: number of output masks
#         :param shuffle: True to shuffle label indexes after every epoch
#         """
#         self.list_IDs = list_IDs
#         self.labels = labels
#         self.image_path = image_path
#         self.mask_path = mask_path
#         self.to_fit = to_fit
#         self.batch_size = batch_size
#         self.dim = dim
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         """Denotes the number of batches per epoch
#         :return: number of batches per epoch
#         """
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#     def __getitem__(self, index):
#         """Generate one batch of data
#         :param index: index of the batch
#         :return: X and y when fitting. X only when predicting
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         X = self._generate_X(list_IDs_temp)
#
#         if self.to_fit:
#             y = self._generate_y(list_IDs_temp)
#             return X, y
#         else:
#             return X
#
#     def on_epoch_end(self):
#         """Updates indexes after each epoch
#         """
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def _generate_X(self, list_IDs_temp):
#         """Generates data containing batch_size images
#         :param list_IDs_temp: list of label ids to load
#         :return: batch of images
#         """
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = self._load_grayscale_image(self.image_path + self.labels[ID])
#
#         return X
#
#     def _generate_y(self, list_IDs_temp):
#         """Generates data containing batch_size masks
#         :param list_IDs_temp: list of label ids to load
#         :return: batch if masks
#         """
#         y = np.empty((self.batch_size, *self.dim), dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])
#
#         return y
#
#     def _load_grayscale_image(self, image_path):
#         """Load grayscale image
#         :param image_path: path to image to load
#         :return: loaded image
#         """
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = img / 255
#         return img


# def val_gen():
#     gt_file = '/storage/dataset/ILSVRC2012/val.txt'
#     val_dir = '/storage/dataset/ILSVRC2012/val/'
#     with open(gt_file, 'r') as f:
#         batch_input = []
#         batch_output = []
#         for line in f:
#             image_name, label = line.strip().split()
#             image = keras.preprocessing.image.load_img(val_dir + image_name, target_size=(224, 224))
#             image = np.array(image, 'float32')
#             image /= 255.
#             image = (image - mean) / std
#             label = keras.utils.to_categorical(int(label), num_classes=1000)
#             batch_input.append(image)
#             batch_output.append(label)
#             if len(batch_input) == BATCH_SIZE:
#                 yield (batch_input, batch_output)
#                 batch_input.clear()
#                 batch_output.clear()


model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=keras.optimizers.SGD(0.1, 0.9, nesterov=True))

START_LR = 0.1
BASE_LR = START_LR * (BATCH_SIZE / 256.0)


def scheduler(epoch):
    if epoch < 30:
        return min(START_LR, BASE_LR)
    elif epoch < 60:
        return BASE_LR * 1e-1
    elif epoch < 90:
        return BASE_LR * 1e-2
    elif epoch < 100:
        return BASE_LR * 1e-3
    else:
        return BASE_LR * 1e-4


change_lr = LearningRateScheduler(scheduler)
tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
# ckpt_cb = ModelCheckpoint(filepath=str(log_dir / '{epoch:02d}.hdf5'), monitor='val_acc', save_weights_only=True,
#                           period=50)
callbacks = [change_lr, tb_cb]

EPOCHS = 150

model.fit_generator(train_data,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    steps_per_epoch=1281167 // BATCH_SIZE,
                    # validation_data=val_gen
                    )

# model.evaluate_generator(test_data)
model.save(log_dir)

# class ResNet18Quantized():
#     def __init__(self, mode, config, model_dir):
#         """
#         mode: Either "training" or "inference"
#         config: A Sub-class of the Config class
#         model_dir: Directory to save training logs and trained weights
#         """
#         self.log_dir = '/home/zhang/blueoil/lmnet/lmnet/networks/instance_segmentation/logs'
#         assert mode in ['training', 'inference']
#         self.mode = mode
#         self.config = config
#         self.model_dir = model_dir
#         self.keras_model = self.build(mode=mode, config=config)
#
#     def build(self, mode, config):
#         """Build Mask R-CNN architecture.
#             input_shape: The shape of the input image.
#             mode: Either "training" or "inference". The inputs and
#                 outputs of the model differ accordingly.
#         """
#         assert mode in ['training', 'inference']
#
#         # Image size must be dividable by 2 multiple times
#         h, w = config.IMAGE_SHAPE[:2]
#         if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
#             raise Exception("Image size must be dividable by 2 at least 6 times "
#                             "to avoid fractions when downscaling and upscaling."
#                             "For example, use 256, 320, 384, 448, 512, ... etc. ")
#
#         # Inputs
#         input_image = KL.Input(
#             shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
#
#         input_gt_class_ids = KL.Input(
#             shape=[None], name="input_gt_class_ids", dtype=tf.int32)
#
#         x, _, _, _, _, _ = resnet_graph_18(input_image, train_bn=config.TRAIN_BN, config=config)
#
#         # x = KL.AveragePooling2D((7, 7), name='avg_pool')(x)
#         x = KL.GlobalAveragePooling2D(name='global_avg_pool')(x)
#         x = KL.Dense(1000, activation='softmax', name='fc1000')(x)
#
#         model = keras.Model(input_image, x, name='resnet18')
#
#         return model
#
#     def compile(self):
#         optimizer = keras.optimizers.SGD(
#             lr=self.config.LEARNING_RATE, momentum=self.config.LEARNING_MOMENTUM,
#             clipnorm=self.config.GRADIENT_CLIP_NORM)
#
#         self.keras_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
#
#     def train(self, train_dataset, val_dataset, epochs, augmentation=None, custom_callbacks=None):
#         # Data generators
#         train_generator = data_generator(train_dataset, self.config, shuffle=True,
#                                          augmentation=augmentation,
#                                          batch_size=self.config.BATCH_SIZE)
#         val_generator = data_generator(val_dataset, self.config, shuffle=True,
#                                        batch_size=self.config.BATCH_SIZE)
#
#         # Create log_dir if it does not exist
#         if not os.path.exists(self.log_dir):
#             os.makedirs(self.log_dir)
#
#         # Callbacks
#         callbacks = [
#             keras.callbacks.TensorBoard(log_dir=self.log_dir,
#                                         histogram_freq=0, write_graph=True, write_images=False),
#             keras.callbacks.ModelCheckpoint(self.checkpoint_path,
#                                             verbose=0, save_weights_only=True, period=50),
#         ]
#
#         # Add custom callbacks to the list
#         if custom_callbacks:
#             callbacks += custom_callbacks
#
#         self.compile()
#
#         self.keras_model.fit_generator(
#             train_generator,
#             initial_epoch=self.epoch,
#             epochs=epochs,
#             steps_per_epoch=self.config.STEPS_PER_EPOCH,
#             callbacks=callbacks,
#             validation_data=val_generator,
#             validation_steps=self.config.VALIDATION_STEPS,
#             max_queue_size=100,
#         )
#         self.epoch = max(self.epoch, epochs)
#
#
# if __name__=='__main__':
#     config = EasyDict()
#     config.IMAGE_SHAPE=[224,224,3]
#     config.TRAIN_BN=True
#     config.LEARNING_RATE = 0.001
#     config.LEARNING_MOMENTUM = 0.9
#     config.WEIGHT_DECAY = 0.0001
#     config.GRADIENT_CLIP_NORM = 5.0
#     config.BATCH_SIZE=48
#     config.STEPS_PER_EPOCH=500 #train dataset num: 1281167
#     config.VALIDATION_STEPS=50*config.STEPS_PER_EPOCH
#     resnet=ResNet18Quantized('training',config,None)

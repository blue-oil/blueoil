Data Augmentation
======

#### Blur

Gaussian blur filter.

    Reference:
        http://pillow.readthedocs.io/en/4.3.x/Referenceserence/ImageFilter.html#PIL.ImageFilter.GaussianBlur

_params_:

`value (int | list | tuple)`: Blur radius. Default is random number from 0 to 1. References default is 2.



#### Brightness

Adjust image brightness.

    Reference:
        http://pillow.readthedocs.io/en/4.2.x/Referenceserence/ImageEnhance.html#PIL.ImageEnhance.Brightness

_params_:

`value (int | list | tuple)`: An enhancement factor of 0.0 gives a black image.
            A factor of 1.0 gives the original image.



#### Color

Adjust image color.

    Reference:
        http://pillow.readthedocs.io/en/4.2.x/Referenceserence/ImageEnhance.html#PIL.ImageEnhance.Color

_params_:

`value (int | list | tuple)`: An enhancement factor of 0.0 gives a black and white image.
            A factor of 1.0 gives the original image.



#### Contrast

Adjust image contrast.

    Reference:
        http://pillow.readthedocs.io/en/4.2.x/Referenceserence/ImageEnhance.html#PIL.ImageEnhance.Contrast

_params_:

`value (int | list | tuple)`: An enhancement factor of 0.0 gives a solid grey image.
            A factor of 1.0 gives the original image.



#### Crop

Crop image.

_params_:

`resize (int | list | tuple)`: If there are resize param, resize and crop.



#### FlipLeftRight

Flip left right.

_params_:

`probability (number)`: Probability for flipping.



#### FlipTopBottom

Flip top bottom.

_params_:

`probability (number)`: Probability for flipping.



#### Hue

Change image hue.

_params_:

`value (int | list | tuple)`: Assume the value in -255, 255. When the value is 0, nothing to do.



#### Pad

Add padding to images.

_params_:

`fill (int)`: Pixel fill value. Default is 0.



#### RandomErasing

Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    The following

_params_:

`sl (float)`: min erasing area

`sh (float)`: max erasing area

`r1 (float)`: min aspect ratio

`content_type (string)`: type of erasing value: {"mean", "random"}

`mean (list)`: erasing value if you use "mean" mode (mean ImageNet pixel value)



#### RandomErasingForDetection

Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    The following

_params_:

`sl (float)`: min erasing area

`sh (float)`: max erasing area

`r1 (float)`: min aspect ratio

`content_type (string)`: type of erasing value: {"mean", "random"}

`mean (list)`: erasing value if you use "mean" mode (mean ImageNet pixel value)

`i_a (bool)`: image-aware, random erase an entire image.

`o_a (bool)`: object-aware, random erase each object bounding boxes.



#### RandomPatchCut

Cut out random patches of the image.

_params_:

`max_size (int)`: maximum size of the patch edge, in percentages of image size

`square (bool)`: force square aspect ratio for patch shape



#### Rotate

Rotate.

_params_:

`angle_range (int | list | tuple)`: Angle range.



#### SSDRandomCrop

SSD random crop.

    References:
        https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py#L208

_params_:

`min_crop_ratio (number)`: Minimum crop ratio for cropping the




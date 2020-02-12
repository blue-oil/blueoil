import numpy as np
import PIL.Image


def load_image(filename, convert_rgb=True):
    """Returns numpy array of an image"""
    image = PIL.Image.open(filename)

    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
    else:
        image = image.convert("L")

    image = np.array(image)

    return image

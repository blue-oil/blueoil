import os
import re
import sys
import copy
import glob
import tqdm
import imageio
import functools
import argparse
import numpy as np
import tensorflow as tf

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

import matplotlib.pyplot as plt
from lmnet.datasets.base import Base
from lmnet.datasets.optical_flow_estimation import FlyingChairs, ChairsSDHom
from lmnet import environment


if __name__ == "__main__":
    print(environment.DATA_DIR)

    train_dataset = FlyingChairs(subset="train")
    validation_dataset = FlyingChairs(subset="validation")
    print(len(train_dataset), len(validation_dataset))

    train_dataset = ChairsSDHom(subset="train")
    validation_dataset = ChairsSDHom(subset="validation")
    print(len(train_dataset), len(validation_dataset))

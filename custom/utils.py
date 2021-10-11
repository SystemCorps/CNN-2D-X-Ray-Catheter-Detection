import cv2
import os
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input

import tensorflow as tf
import gc
from glob import glob
import imgaug as ia
import imgaug.augmenters as iaa


class segDataLoaderSingle(keras.utils.Sequence):
    def __init__(self, p, s, root, h=256, w=256, test_size=0.2):
        self.patient = p
        self.set = s
        self.h = h
        self.w = w
        self.test_size = test_size

        self.ps_root = os.path.join(root, '%02d' % self.p)
        self.ps_root = os.path.join(self.ps_root, 'set%02d' % self.s)
        self.flus = glob(os.path.join(self.ps_root, 'fluoroReal/*png'))
        self.flus.sort()

        self.
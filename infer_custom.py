import numpy as np
import os
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

import imgaug
from sklearn.model_selection import train_test_split

from utils_dh import utils
from python.common.NnetsX import NNets ,MyReLU

import cv2
import math

import datetime

gpu = 0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu], True)
    except RuntimeError as e:
        print(e)


model_name = '../FluSeg/models/1011_095705/model_best.h5'

infer_dir = os.path.split(model_name)[0].replace('models', 'infer')

if not os.path.exists(infer_dir):
    os.makedirs(infer_dir)

x_dir = '/docker/i2i/EntireData/05/set01/fluoroReal'
y_dir = '/docker/i2i/EntireData/05/set01/fluoroAll_DN'

x_dirs = glob(os.path.join(x_dir, '*.png'))
y_dirs = glob(os.path.join(y_dir, '*.png'))
x_dirs.sort()
y_dirs.sort()

x_train, x_valid, y_train, y_valid = train_test_split(x_dirs, y_dirs, test_size=0.2)

hyper = {'Width':256, 'Height':256}

valid_data = utils.dataLoading(x_valid, y_valid, hyper, list_given=True)

BATCH_NORMALIZATION_NO = -1
BATCH_NORMALIZATION_YES = 0

DOWNSAMPLING_MAXPOOL = 0
DOWNSAMPLING_STRIDED_CONV = 1

UPSAMPLING_UPSAMPLE = 0
UPSAMPLING_TRANSPOSED_CONV = 1

batch_size = 8
epochs = 300
n_channel = 4
input_shape = (256,256,n_channel)

nnets = NNets()
nnets.m_SamePartActivation = MyReLU
nnets.m_RegularizerL1L2 = False
nnets.m_Dropout = 0.5
nnets.m_Residual = True
nnets.m_BatchNormalization = BATCH_NORMALIZATION_YES
nnets.m_BorderMode = "same"
nnets.m_Initialization = "glorot_uniform"
nnets.m_DownSampling = DOWNSAMPLING_STRIDED_CONV
nnets.m_UpSampling = UPSAMPLING_UPSAMPLE
nbUsedChannel = n_channel
nbStartFilter = 8
kernelSize = 3
nbConvPerLayer = [2, 2, 2, 2, 2, 2, 2]
nbDeconvPerLayer = [2, 2, 2, 2, 2, 2]

optimizer = SGD(lr=0.01, decay=5e-4, momentum=0.99)

best_model = nnets.DefineDeepUVNet(input_shape, _nbFilters=nbStartFilter, _kernelSize=kernelSize,
                                   _convPerLevel=nbConvPerLayer, _upConvPerLevel=nbDeconvPerLayer, _optimizer=optimizer)

best_model.load_weights(model_name)

preds = best_model.predict(valid_data.X)

for i in range(len(preds)):
    x_img = valid_data.X[i,...,0]
    if x_img.shape[-1] == 1:
        h, w, _ = x_img.shape
        x_img = x_img.reshape((h, w))
    x_img = x_img * 255
    x_img = x_img.astype(np.uint8)

    y_img = preds[i]
    if y_img.shape[-1] == 1:
        h, w, _ = y_img.shape
        y_img = y_img.reshape((h, w))
    y_img = y_img * 255
    y_img = y_img.astype(np.uint8)
    x_name = os.path.join(infer_dir, 'x_%05d.png' % i)
    y_name = os.path.join(infer_dir, 'y_%05d.png' % i)

    cv2.imwrite(x_name, x_img)
    cv2.imwrite(y_name, y_img)

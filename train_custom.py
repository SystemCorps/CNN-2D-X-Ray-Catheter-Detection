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

tf.keras.backend.set_image_data_format('channels_last')

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

save_dir = '../FluSeg'

now = datetime.datetime.now()
trial_date = "{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.month, now.day, now.hour, now.minute, now.second)

model_dir = os.path.join(save_dir, 'models/{}'.format(trial_date))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_name = os.path.join(model_dir, 'model_best.h5')

sum_dir = os.path.join(save_dir, 'summary/{}'.format(trial_date))
if not os.path.exists(sum_dir):
    os.makedirs(sum_dir)

infer_dir = os.path.join(save_dir, 'infer/{}'.format(trial_date))
if not os.path.exists(infer_dir):
    os.makedirs(infer_dir)


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

model = nnets.DefineDeepUVNet(input_shape, _nbFilters=nbStartFilter, _kernelSize=kernelSize,
                              _convPerLevel=nbConvPerLayer, _upConvPerLevel=nbDeconvPerLayer, _optimizer=optimizer)

print(model.summary())

x_dir = '/docker/i2i/EntireData/05/set01/fluoroReal'
y_dir = '/docker/i2i/EntireData/05/set01/fluoroAll_DN'

x_dirs = glob(os.path.join(x_dir, '*.png'))
y_dirs = glob(os.path.join(y_dir, '*.png'))
x_dirs.sort()
y_dirs.sort()

x_train, x_valid, y_train, y_valid = train_test_split(x_dirs, y_dirs, test_size=0.2)

hyper = {'Width':256, 'Height':256}

train_data = utils.dataLoading(x_train, y_train, hyper, list_given=True)
valid_data = utils.dataLoading(x_valid, y_valid, hyper, list_given=True)

data_gen_args = dict(rotation_range=9,
                     width_shift_range=0.16,
                     height_shift_range=0.16,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='constant',
                     cval=0,
                     zoom_range=0.1)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed = 5920
image_datagen.fit(train_data.X, augment=True, seed=seed)
mask_datagen.fit(train_data.Y, augment=True, seed=seed)

x_train_gen = image_datagen.flow(train_data.X, seed=seed, batch_size=batch_size)
y_train_gen = mask_datagen.flow(train_data.Y, seed=seed, batch_size=batch_size)

train_gen = zip(x_train_gen, y_train_gen)

valid_gen = utils.valDataGen(x_valid, y_valid, hyper, batch_size, list_given=True)


tbcallback = TensorBoard(log_dir=sum_dir)
checkpoint = ModelCheckpoint(filepath=model_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
callbacks = [tbcallback, checkpoint]

steps_per_epoch = math.ceil(len(x_train)//batch_size)

model.fit(train_gen,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          steps_per_epoch=steps_per_epoch,
          validation_data=valid_gen,
          callbacks=callbacks,
          workers=6,
          max_queue_size=100)


del model

best_model = keras.models.load_model(model_name)

preds = best_model.predict(valid_data.X)

for i in range(len(preds)):
    x_img = valid_data.X[i, ..., 0]
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

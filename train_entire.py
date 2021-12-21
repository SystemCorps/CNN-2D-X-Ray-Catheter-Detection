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
import pickle as pk


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

save_dir = '../FluSegTip_entire'

now = datetime.datetime.now()
trial_date = "{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.month, now.day, now.hour, now.minute, now.second)

model_dir = os.path.join(save_dir, 'models/{}'.format(trial_date))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_name = os.path.join(model_dir, 'model_best.h5')

sum_dir = os.path.join(save_dir, 'summary/{}'.format(trial_date))
if not os.path.exists(sum_dir):
    os.makedirs(sum_dir)
hyper_name = sum_dir + '.txt'

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

scale_add = 0.1
trans = 0.16
rot = 9.0
cval = 0.0
fprob = 0.5

scale = {"x": (1.-scale_add, 1.+scale_add), "y": (1.-scale_add, 1.+scale_add)}
translation = {"x": (-trans, trans), "y": (-trans, trans)}
rotate = (-rot, rot) # Degree

ps_dict_root = '/docker/entire_list_1011.pkl'
ps_dict = pk.load(open(ps_dict_root, 'rb'))
ps_root = '/docker/i2i/EntireData'

hyper = {'Width':256, 'Height':256,
         'FlipProb':fprob,
         'AffineScale':scale,
         'AffineTrans':translation,
         'AffineRot':rotate,
         'AffineCval': cval,
         'Dataset':[ps_dict_root, ps_root],
         'ModelSavingPath': model_name,
         'SummaryDir': sum_dir,
         'BatchSize':batch_size,
         'Epochs':epochs}


train_gen = utils.entireDataGen(ps_dict['train'], ps_root, hyper, batch_size=batch_size, tipOnly=True)
valid_gen = utils.entireDataGen(ps_dict['valid'], ps_root, hyper, batch_size=batch_size, valid=True, tipOnly=True)

tbcallback = TensorBoard(log_dir=sum_dir)
checkpoint = ModelCheckpoint(filepath=model_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
callbacks = [tbcallback, checkpoint]

steps_per_epoch = math.ceil(len(train_gen.x_lists)//batch_size)

model.fit(train_gen,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          steps_per_epoch=steps_per_epoch,
          validation_data=valid_gen,
          callbacks=callbacks,
          workers=6,
          max_queue_size=100)
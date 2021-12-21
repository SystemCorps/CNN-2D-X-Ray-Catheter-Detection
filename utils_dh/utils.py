import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from glob import glob
import math
import random


def loadImgs(img_dirs, h=256, w=256, norm=True, axis=-1):
    outputs = None

    for path in img_dirs:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (w, h))
        if norm:
            img = img / 255.0
        else:
            img = img / 1.0

        if outputs is None:
            if axis == -1:
                outputs = img.reshape((h, w, 1))
            elif axis == 0:
                outputs = img.reshape((1, h, w, 1))
        else:
            if axis == -1:
                outputs = np.concatenate((outputs, img.reshape(h, w, 1)), axis=axis)
            elif axis == 0:
                outputs = np.concatenate((outputs, img.reshape(1, h, w, 1)), axis=axis)

    return outputs.astype(np.float32)


def inputAlign(imgs, h, w):
    length = imgs.shape[-1]

    X = np.zeros((length, h, w, 4))

    for i in range(length):
        seq = np.zeros((h, w, 4))

        for j in range(seq.shape[-1]):
            if i - j >= 0:
                seq[..., j] = imgs[..., i - j]
        X[i, ...] = seq

    return X


def inputAlignDirs(dirs):
    aligned = []

    for i in range(len(dirs)):
        seq = [None] * 4
        for j in range(len(seq)):
            if i - j >= 0:
                seq[j] = dirs[i-j]
        aligned.append(seq)
    return aligned


def loadAlignedBatch(x_batch, h=256, w=256):

    outputs = np.zeros((len(x_batch), h, w, 4), np.float32)

    for i in range(len(x_batch)):
        dirs = x_batch[i]
        temp = None
        for path in dirs:
            if path is None:
                if temp is None:
                    temp = np.zeros((h,w,1))
                else:
                    temp = np.concatenate((temp, np.zeros((h,w,1))), axis=-1)
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (w,h)) / 255.0
            img = img.reshape((h,w,1))

            if temp is None:
                temp = img
            else:
                temp = np.concatenate((temp, img), axis=-1)

        outputs[i,...] = temp

    return outputs


class entireDataGen(tf.keras.utils.Sequence):
    def __init__(self, ps_dict, ps_root, hyper, batch_size, valid=False, tipOnly=True, target_p=None, target_s=None, shuffle=True):
        self.valid = valid
        self.x_lists = []
        self.y_lists = []
        self.hyper = hyper
        self.batch_size = batch_size
        self.shuffle = shuffle
        if target_p is None and target_s is None:
            for p in ps_dict.keys():
                for s in ps_dict[p]:
                    if isinstance(p, int):
                        pd = '%02d' % p
                    else:
                        pd = p
                    if isinstance(s, int):
                        sd = 'set%02d' % s
                    else:
                        sd = s
                    root = os.path.join(ps_root, pd)
                    root = os.path.join(root, sd)
                    x_temp = glob(os.path.join(root, 'fluoroReal/*.png'))
                    x_temp.sort()
                    x_temp = inputAlignDirs(x_temp)
                    if tipOnly:
                        y_temp = glob(os.path.join(root, 'fluoroTip_DN/*.png'))
                    else:
                        y_temp = glob(os.path.join(root, 'fluoroAll_DN/*.png'))
                    y_temp.sort()
                    if len(x_temp) == 0 or len(y_temp) == 0:
                        print(root)
                        continue
                    if len(x_temp) != len(y_temp):
                        print(root)
                        continue
                    self.x_lists.extend(x_temp)
                    self.y_lists.extend(y_temp)
        else:
            if isinstance(target_p, int):
                pd = '%02d' % target_p
            else:
                pd = target_p
            if isinstance(target_s, int):
                sd = 'set%02d' % target_s
            else:
                sd = target_s
            root = os.path.join(ps_root, pd)
            root = os.path.join(root, sd)
            x_temp = glob(os.path.join(root, 'fluoroReal/*.png'))
            x_temp.sort()
            x_temp = inputAlignDirs(x_temp)
            if tipOnly:
                y_temp = glob(os.path.join(root, 'fluoroTip_DN/*.png'))
            else:
                y_temp = glob(os.path.join(root, 'fluoroAll_DN/*.png'))
            y_temp.sort()
            self.x_lists.extend(x_temp)
            self.y_lists.extend(y_temp)
            
        seed = 80433
        if shuffle:
            random.Random(seed).shuffle(self.x_lists)
            random.Random(seed).shuffle(self.y_lists)

        self.batch_size = batch_size

        self.w = hyper['Width']
        self.h = hyper['Height']

        self.flip_prob = hyper['FlipProb']
        self.scale = hyper['AffineScale']
        self.trans = hyper['AffineTrans']
        self.rot = hyper['AffineRot']
        self.cval = hyper['AffineCval']
        self.affine = iaa.Affine(scale=self.scale,
                                 translate_percent=self.trans,
                                 rotate=self.rot,
                                 cval=self.cval,
                                 mode='constant')
        self.seq = iaa.Sequential([iaa.Fliplr(self.flip_prob),
                                   iaa.Flipud(self.flip_prob),
                                   self.affine])

    def __len__(self):
        return math.ceil(len(self.x_lists) / self.batch_size)

    """data_gen_args = dict(rotation_range=9,
                         width_shift_range=0.16,
                         height_shift_range=0.16,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='constant',
                         cval=0,
                         zoom_range=0.1)"""

    def __getitem__(self, index):
        x_list_batch = self.x_lists[index * self.batch_size:(index+1) * self.batch_size]
        y_list_batch = self.y_lists[index * self.batch_size:(index + 1) * self.batch_size]

        x_raws = loadAlignedBatch(x_list_batch, h=self.h, w=self.w).astype(np.float32)
        y_raws = loadImgs(y_list_batch, norm=False, axis=0).astype(np.float32)

        #x_raws_aligned = inputAlign(x_raws, self.h, self.w)

        if self.valid:
            return x_raws, y_raws
        else:
            x_batch, y_batch = self.seq(images=x_raws, heatmaps=y_raws)
            return x_batch, y_batch


class valDataGen(tf.keras.utils.Sequence):
    def __init__(self, X_dir, Y_dir, hyper, batch_size, list_given=False):
        if not list_given:
            self.X_dirs = glob(os.path.join(X_dir, '*.png'))
            self.X_dirs.sort()

            self.Y_dirs = glob(os.path.join(Y_dir, '*.png'))
            self.Y_dirs.sort()
        else:
            self.X_dirs = X_dir
            self.Y_dirs = Y_dir

        self.batch_size = batch_size

        self.w = hyper['Width']
        self.h = hyper['Height']
        self.X_imgs_temp = loadImgs(self.X_dirs)
        self.X = inputAlign(self.X_imgs_temp, self.h, self.w)
        del self.X_imgs_temp
        self.Y = loadImgs(self.Y_dirs, norm=False, axis=0)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        x_batch = self.X[index * self.batch_size:(index+1) * self.batch_size]
        y_batch = self.Y[index * self.batch_size:(index+1) * self.batch_size]

        return x_batch, y_batch


class dataLoading():
    def __init__(self, X_dir, Y_dir, hyper, list_given=False):
        if not list_given:
            self.X_dirs = glob(os.path.join(X_dir, '*.png'))
            self.X_dirs.sort()

            self.Y_dirs = glob(os.path.join(Y_dir, '*.png'))
            self.Y_dirs.sort()
        else:
            self.X_dirs = X_dir
            self.Y_dirs = Y_dir

        self.w = hyper['Width']
        self.h = hyper['Height']
        self.X_imgs_temp = loadImgs(self.X_dirs)
        self.X = inputAlign(self.X_imgs_temp, self.h, self.w)
        del self.X_imgs_temp
        self.Y = loadImgs(self.Y_dirs, norm=False, axis=0)

        # self.X_train, self.Y_train, self.X_valid, self.Y_valid = train_test_split(self.X_dirs, self.Y_dirs, test_size=0.2)




from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, concatenate, Activation, Dense, Reshape, Lambda, Dropout, Multiply, Add
#from tensorflow.keras.layers import RandomTranslation, RandomRotation, RandomZoom
from tensorflow.keras.layers import SpatialDropout2D

from tensorflow.keras import models


import numpy as np
import tensorflow as tf
import os
import json
from glob import glob

def unet_renew(input_shape, aug_param, layers=4, filter_root=64, kernel_size=3, pool_size=2, output_channel=3, fixed_angio=None,
               conv_act='relu', k_init="glorot_uniform", b_init="zeros"):

    img_aug = keras.Sequential([RandomTranslation(aug_param['trans']['x'], aug_param['trans']['y'], fill_value=1.0, fill_mode='constant'),
                                RandomRotation(aug_param['rot'], fill_value=1.0, fill_mode='constant'),
                                RandomZoom(aug_param['scale'], fill_value=1.0, fill_mode='constant')])

    inputs = Input(shape=input_shape)

    def operateWithConstant(input_batch):
        tf_constant = K.constant(fixed_angio)
        batch_size = K.shape(input_batch)[0]
        tiled_constant = K.tile(tf_constant, (batch_size, 1, 1, 1))
        return tiled_constant

    def runImageAug(input):
        first = tf.expand_dims(input[..., 0], axis=-1)
        angios = input[..., 1:]
        augmented = img_aug(first)
        return concatenate([augmented, angios], axis=-1)

    if fixed_angio is not None:
        fixed = Lambda(operateWithConstant)(inputs)
        input_tensor = img_aug(inputs)
        input_tensor = concatenate([input_tensor, fixed], axis=-1)

        h, w, c = input_shape
        num_angio = K.shape(fixed_angio)[-1]
        base_unet = unet((h, w, c + num_angio), layers, filter_root, kernel_size, pool_size, output_channel,
                         conv_act=conv_act)

    else:
        input_tensor = Lambda(runImageAug)(inputs)
        base_unet = unet(input_shape, layers, filter_root, kernel_size, pool_size, output_channel,
                         conv_act=conv_act, k_init=k_init, b_init=b_init)

    outs = base_unet(input_tensor)

    model = models.Model(inputs, outs)
    return model



def unet(input_shape, layers=4, filter_root=64, kernel_size=3, pool_size=2, output_channel=3,
         kernel_l2_factor=0.0, bias_l2_factor=0.0,fixed_angio=None, conv_act='relu', residual=False,
         k_init="glorot_uniform", b_init="zeros",
         drop_rate=0.0):
    if conv_act == 'prelu':
        conv_act = tf.keras.layers.PReLU()
    if conv_act == 'leaky':
        conv_act = tf.keras.layers.LeakyReLU(alpha=0.01)
    
    
    inputs = Input(shape=input_shape)
    # Spatial-channel attention https://www.frontiersin.org/articles/10.3389/fbioe.2020.00670/full
    #input_tensor = models.Input(shape=input_shape)
    def operateWithConstant(input_batch):
        tf_constant = K.constant(fixed_angio)
        batch_size = K.shape(input_batch)[0]
        tiled_constant = K.tile(tf_constant, (batch_size, 1,1,1))
    # Do some operation with tiled_constant and input_batch
        return tiled_constant
    
    
    if fixed_angio is not None:
        fixed = Lambda(operateWithConstant)(inputs)
        input_tensor = concatenate([inputs, fixed], axis=-1)
    
    else:
        input_tensor = inputs
    
    
    conv = []
    
    if kernel_l2_factor:
        kr = keras.regularizers.l2(kernel_l2_factor)
    else:
        kr = None
    
    if bias_l2_factor:
        br = keras.regularizers.l2(bias_l2_factor)
    else:
        br = None
    # for 6 layers, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(layers):
        num_filters = (2**i) * filter_root
        conv1 = Conv2D(num_filters, kernel_size, padding='same', activation=conv_act,
                       kernel_regularizer=kr, bias_regularizer=br,
                       kernel_initializer=k_init, bias_initializer=b_init)(input_tensor)
        conv2 = Conv2D(num_filters, kernel_size, padding='same', activation=conv_act,
                       kernel_regularizer=kr, bias_regularizer=br,
                       kernel_initializer=k_init, bias_initializer=b_init)(conv1)

        if drop_rate:
            conv2 = SpatialDropout2D(rate=drop_rate)(conv2)
        
        if residual:
            shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(input_tensor)
            conv2 = Add()([conv2, shortcut])

        conv.append(conv2)
        
        if i < layers-1:
            input_tensor = MaxPooling2D(pool_size=pool_size, padding='same')(conv2)
            
    input_tensor = conv[-1]
    
    for i in range(layers-2, -1, -1):
        num_filters = (2**(i+1)) * filter_root
        
        deconv = Conv2DTranspose(num_filters//2, kernel_size=pool_size, strides=pool_size,
                                 padding='same', activation=conv_act, kernel_regularizer=kr, bias_regularizer=br,
                                 kernel_initializer=k_init, bias_initializer=b_init)(input_tensor)
        deconv_concat = concatenate([conv[i], deconv], axis=3)
        
        conv1 = Conv2D(num_filters//2, kernel_size=kernel_size, padding='same', activation=conv_act,
                       kernel_regularizer=kr, bias_regularizer=br,
                       kernel_initializer=k_init, bias_initializer=b_init)(deconv_concat)
        
        conv2 = Conv2D(num_filters//2, kernel_size=kernel_size, padding='same', activation=conv_act,
                       kernel_regularizer=kr, bias_regularizer=br,
                       kernel_initializer=k_init, bias_initializer=b_init)(conv1)

        if drop_rate:
            conv2 = SpatialDropout2D(rate=drop_rate)(conv2)
        
        if residual:
            shortcut = Conv2D(num_filters//2, kernel_size=(1, 1), padding='same',
                              kernel_initializer=k_init, bias_initializer=b_init)(deconv_concat)
            input_tensor = Add()([conv2, shortcut])
        else:
            input_tensor = conv2
        
        
    outputs = Conv2D(output_channel, kernel_size, padding='same', activation=None)(input_tensor)
    outputs = Activation('tanh')(outputs)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def resUnet(input_shape, layers=4, filter_root=64, kernel_size=3, pool_size=2, output_channel=3, kernel_l2_factor=None, bias_l2_factor=None, bn=False):
    
    def bn_act(x, act=True):
        x = keras.layers.BatchNormalization()(x)
        if act == True:
            x = keras.layers.Activation("relu")(x)
        return x

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1, bn=True):
        if bn:
            conv = bn_act(x)
        else:
            conv = Activation('relu')(x)
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1, bn=True):
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides, bn=bn)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        if bn:
            shortcut = bn_act(shortcut, act=False)
        output = keras.layers.Add()([conv, shortcut])
        return output

    def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1, bn=True):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides, bn=bn)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1, bn=bn)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        if bn:
            shortcut = bn_act(shortcut, act=False)

        output = keras.layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip, num_filters):
        #u = keras.layers.UpSampling2D((2, 2))(x)
        u = Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding='same', activation=None)(x)
        c = keras.layers.Concatenate()([u, xskip])
        return c
    
    
    filters = []
    for i in range(layers):
        num_filter = filter_root * 2 ** i
        filters.append(num_filter)
        
    inputs = Input(shape=input_shape)
    input_tensor = inputs
    
    encoders = []
    for i in range(layers):
        if i == 0:
            enc = stem(input_tensor, filters[i], bn=bn)
        else:
            enc = conv_block(encoders[-1], filters[i], strides=2, bn=bn)
        encoders.append(enc)
    
    b0 = conv_block(encoders[-1], filters[-1], strides=1, bn=bn)
    b1 = conv_block(b0, filters[-1], strides=1, bn=bn)
    
    decoders = []
    # 64 128 256 512 1024 2048
    for i in range(layers-2, -1, -1):
        if i == layers-2:
            up = upsample_concat_block(b1, encoders[i], filters[i])
        else:
            up = upsample_concat_block(dec, encoders[i], filters[i])
        dec = residual_block(up, filters[i], bn=bn)
    
    outputs = Conv2D(output_channel, kernel_size, padding='same', activation=None)(dec)
    outputs = Activation('tanh')(outputs)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model



def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator(input_shape, kernel_size=4, output_channel=3):
    inputs = tf.keras.layers.Input(shape=input_shape)

    down_stack = [
    downsample(64, kernel_size, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, kernel_size), # (bs, 64, 64, 128)
    downsample(256, kernel_size), # (bs, 32, 32, 256)
    downsample(512, kernel_size), # (bs, 16, 16, 512)
    downsample(512, kernel_size), # (bs, 8, 8, 512)
    downsample(512, kernel_size), # (bs, 4, 4, 512)
    downsample(512, kernel_size), # (bs, 2, 2, 512)
    downsample(512, kernel_size), # (bs, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, kernel_size, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, kernel_size, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, kernel_size, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, kernel_size), # (bs, 16, 16, 1024)
    upsample(256, kernel_size), # (bs, 32, 32, 512)
    upsample(128, kernel_size), # (bs, 64, 64, 256)
    upsample(64, kernel_size), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channel, kernel_size,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator(input_shape, input_shape_2=None):
    initializer = tf.random_normal_initializer(0., 0.02)

    if input_shape_2 is None:
        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=input_shape, name='target_image')
    else:
        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=input_shape_2, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import losses
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import UpSampling2D

class Conv2DBlock(layers.Layer):

    """
    A general layer to abstract out some of the details of the Downsample blocks.
    Runs multiple conv2d layers, along with their activation layers, in parallel.
    """

    def __init__(self,
                 filters=2,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 shape=[256,1,1],
                 **kwargs):

        #Lets it inherit from Layer if we need that functionality in the future.
        super(Conv2DBlock, self).__init__(**kwargs)

        self.conv2d_1 = Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                input_shape=shape)
        self.activation_1 = layers.Activation("relu")

        self.conv2d_2 = Conv2D(filters=filters,
                                kernel_size=(3,3),
                                strides=(1,1),
                                padding="same",
                                input_shape=shape)
        self.activation_2 = layers.Activation("relu")

    #Turns this class into a callable, so we can pass the input in directly on creation of the object.
    def __call__(self, inputs, **kwargs):
        x = inputs
        x = self.conv2d_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)
        x = self.activation_2(x)
        return x

class Upconv2DBlock(layers.Layer):
    """
    A general layer to abstract out some of the details of the Upsample blocks.
    Runs multiple upconv layers, along with their activation layers, in parallel.
    """
    def __init__(self,
                 filters=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="same",
                 **kwargs):
        super(Upconv2DBlock, self).__init__(**kwargs)

        self.upconv_1 = layers.Conv2DTranspose(filters=filters // 2,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding)
        self.activation_1 = layers.Activation("relu")

        self.upconv_2 = layers.Conv2DTranspose(filters=filters // 2,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding)

        self.activation_2 = layers.Activation("relu")

    #Makes this class a callable, so we can pass the input in directly.
    def __call__(self, inputs, **kwargs):
        x = inputs
        x = self.upconv_1(x)
        x = self.activation_1(x)
        x = self.upconv_2(x)
        x = self.activation_2(x)
        return x

class ConcatBlock(layers.Layer):

    """
    A simple callable class to concatenate layers together.
    """

    def __call__(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = max((x1_shape[1] - x2_shape[1]), x2_shape[1] - x1_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x
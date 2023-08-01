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
                                shape=shape)
        self.activation_1 = layers.Activation("relu")

        self.conv2d_2 = Conv2D(filters=filters,
                                kernel_size=(3,3),
                                strides=(1,1),
                                padding="same",
                                shape=shape)
        self.activation_2 = layers.Activation("relu")

    #Turns this class into a callable, so we can pass the input in directly on creation of the object.
    def call(self, inputs, **kwargs):
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
                 ** kwargs):
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
    def call(self, inputs, **kwargs):
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

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x





def build_model(filters: int = 2,
                frequency_bins: int = 256,
                time_steps: int = 1,
                channels: int = 1,
                ) -> Model:
    """
    Constructs a U-Net model

    :param filters: the number of filters being used for this model
    :param frequency_bins: one part of the input shape; the number of frequencies we're modelling
    :param time_steps: one part of the input shape; how many time-steps we're modelling for
    :param channels: one part of the input shape: number of channels of the input tensors

    :return: A Keras model, ready to be used
    """

    #Creates an Input object for future uses of this model to be run through.
    inputs = Input(shape=(frequency_bins, time_steps, channels), name="inputs")

    #In this case, x will be the actual object we interface with layers through throughout this call.
    x = inputs
    contracting_layers = {}

    #Hardcoded in due to time constraints.
    ds_layer_shapes = [
        [frequency_bins, time_steps, channels],
        [frequency_bins, time_steps, channels * 4],
        [frequency_bins / 2, time_steps, channels * 4],
        [frequency_bins / 4, time_steps, channels * 4],
        [frequency_bins / 8, time_steps, channels * 8],
        [frequency_bins / 16, time_steps, channels * 8],
        [frequency_bins / 32, time_steps, channels * 16],
    ]

    us_layer_shapes = ds_layer_shapes[::-1]

    #Each use of a layer will update x as we proceed through the function.
    x = Conv2DBlock(filters=filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    shape=ds_layer_shapes[1])(x)

    #Contracting_layers is for passing through the upsample blocks later.
    contracting_layers[1] = x

    #Creates and uses most of the downscaling blocks.
    for layer_id in range(len(ds_layer_shapes) - 2):
        layer_shape = ds_layer_shapes[layer_id + 2]

        x = layers.MaxPooling2D((2, 1))(x)

        x = Conv2DBlock(filters=filters,
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding="same",
                        shape=layer_shape,
                    )(x)

        contracting_layers[layer_id + 2] = x

    #Creates and uses all of the upscaling blocks.
    for layer_id in range(len(us_layer_shapes) - 1):
        layer_shape = us_layer_shapes[layer_id + 1]

        x = UpSampling2D(size=(2,1))
        x = Upconv2DBlock(filters=filters,
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding="same",
                        shape=layer_shape)(x)
        x = ConcatBlock()(x, contracting_layers[len(us_layer_shapes) -layer_id])
        x = Conv2DBlock(filters=filters,
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding="same",
                        shape=layer_shape,
                    )(x)

    #Performs final convolution
    x = layers.Conv2D(filters=filters,
                      kernel_size=(3,3),
                      strides=1,
                      padding="same")(x)

    #Creates final output layer, creates the model, names it, compiles it, and returns it.
    x = layers.Activation("relu")(x)
    outputs = layers.Activation("softmax", name="outputs")(x)
    model = Model(inputs, outputs, name="unet")

    model.compile()
    return model

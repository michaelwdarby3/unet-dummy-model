from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import UpSampling2D

import generic_conv_utils


def build_model(filters: int = 2,
                frequency_bins: int = 256,
                time_steps: int = 1,
                channels: int = 1) -> Model:
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
    x = generic_conv_utils.Conv2DBlock(filters=filters,
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

        x = generic_conv_utils.Conv2DBlock(filters=filters,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding="same",
                                           shape=layer_shape,
                                           )(x)

        contracting_layers[layer_id + 2] = x

    #Creates and uses all of the upscaling blocks.
    for layer_id in range(len(us_layer_shapes) - 1):
        layer_shape = us_layer_shapes[layer_id + 1]

        x = UpSampling2D(size=(2,1))(x)
        x = generic_conv_utils.Upconv2DBlock(filters=filters,
                                             kernel_size=(3,3),
                                             strides=(1,1),
                                             padding="same")(x)
        # x = generic_conv_utils.ConcatBlock()(x, contracting_layers[len(us_layer_shapes) - layer_id - 1])
        x = generic_conv_utils.Conv2DBlock(filters=filters,
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

    return model

def make_model(model: Model):
    model.compile()
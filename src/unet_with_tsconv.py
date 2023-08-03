from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D

import generic_conv_utils


#TSConv is the class defining a timeshifted conv2d block
class TSConv(generic_conv_utils.Conv2DBlock):
    def __init__(self,
                 filters=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="same",
                 shape=[256, 1, 1],
                 **kwargs):
        self.shape = shape
        super(TSConv, self).__init__(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     shape=shape,
                                     **kwargs)

    #Ideally, shifting would be done within the block; however, because of issues with datatypes,
    #   this doesn't seem possible using TensorFlow (barring further research)
    """def shift(self, inputs):
        left_cutoff = self.shape[0] // 4
        right_cutoff = self.shape[0] // 2

        for i in range(self.shape[0]):
            if i <= left_cutoff:
                inputs = [0] + inputs[i][:-1]
            if i > left_cutoff and i <= right_cutoff:
                inputs[i] = inputs[i][1:] + [0]
        return inputs"""

    def __call__(self, inputs):
        #shifted_inputs = self.shift(inputs)
        return generic_conv_utils.Conv2DBlock.call(self, inputs)

class GenericConvBlock(layers.Layer):
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
                 is_ts_conv=True,
                 **kwargs):

        #Lets it inherit from Layer if we need that functionality in the future.
        super(GenericConvBlock, self).__init__(**kwargs)

        if is_ts_conv:
            conv_layer = TSConv
        else:
            conv_layer = Conv2D

        self.convlayer_1 = conv_layer(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                shape=shape)
        self.activation_1 = layers.Activation("relu")

        self.convlayer_2 = conv_layer(filters=filters,
                                kernel_size=(3,3),
                                strides=(1,1),
                                padding="same",
                                shape=shape)
        self.activation_2 = layers.Activation("relu")

    #Turns this class into a callable, so we can pass the input in directly on creation of the object.
    def __call__(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.convlayer_1(x)
        x = self.activation_1(x)
        x = self.convlayer_2(x)
        x = self.activation_2(x)
        return x

def build_model(filters: int = 2,
                frequency_bins: int = 256,
                time_steps: int = 1,
                channels: int = 1,
                is_ts_conv=True,
                ) -> Model:
    """
    Constructs a U-Net model

    :param filters: the number of filters being used for this model
    :param frequency_bins: one part of the input shape; the number of frequencies we're modelling
    :param time_steps: one part of the input shape; how many time-steps we're modelling for
    :param channels: one part of the input shape: number of channels of the input tensors
    :param is_ts_conv: Defines whether to use ts_conv layers.

    :return: A Keras model, ready to be used
    """

    #Creates an Input object for future uses of this model to be run through.
    inputs = Input(shape=(frequency_bins, time_steps, channels), name="inputs")

    # In this case, x will be the actual object we interface with layers through throughout this call.
    x = inputs
    contracting_layers = {}
    if is_ts_conv:
        kernel_size=(3,1)
    else:
        kernel_size=(3,3)

    # Hardcoded in due to time constraints.
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
    x = GenericConvBlock(filters=filters,
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    padding="same",
                    shape=ds_layer_shapes[1],
                    is_ts_conv=is_ts_conv,)(x)

    #Contracting_layers is for passing through the upsample blocks later.
    contracting_layers[1] = x

    #Creates and uses most of the downscaling blocks.
    for layer_id in range(len(ds_layer_shapes) - 2):
        layer_shape = ds_layer_shapes[layer_id + 2]

        x = layers.MaxPooling2D((2, 1))(x)

        x = GenericConvBlock(filters=filters,
                        kernel_size=kernel_size,
                        strides=(1,1),
                        padding="same",
                        shape=layer_shape,
                        is_ts_conv=is_ts_conv,
                    )(x)

        contracting_layers[layer_id + 2] = x

    #Creates and uses all of the upscaling blocks.
    for layer_id in range(len(us_layer_shapes) - 1):
        layer_shape = us_layer_shapes[layer_id + 1]
        x = generic_conv_utils.Upconv2DBlock(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=(1,1),
                                             padding="same")(x)
        x = generic_conv_utils.ConcatBlock()(x, contracting_layers[layer_id + 1])
        x = GenericConvBlock(filters=filters,
                        kernel_size=kernel_size,
                        strides=(1,1),
                        padding="same",
                        shape=layer_shape,
                        is_ts_conv=is_ts_conv,
                    )(x)

    #Performs final convolution
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=1,
                      padding="same")(x)

    #Creates final output layer, creates the model, names it, compiles it, and returns it.
    x = layers.Activation("relu")(x)
    outputs = layers.Activation("softmax", name="outputs")(x)
    model = Model(inputs, outputs, name="unet")

    return model

def make_model(model: Model):
    model.compile()



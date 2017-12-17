# COPYRIGHT

# All contributions by Raghavendra Kotikalapudi:
# Copyright (c) 2016, Raghavendra Kotikalapudi.
# All rights reserved.

# All other contributions:
# Copyright (c) 2016, the respective contributors.
# All rights reserved.

# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.

# LICENSE

# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    TimeDistributed
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def make_time_distributed_simple(model, TIME_STEPS, input_shape):
    time_distributed_model = Sequential()
    for model_layer_index in range(len(model.layers)):
        # print(model_layer_index)
        if model_layer_index == 0:
            time_distributed_model.add(TimeDistributed(model.layers[model_layer_index], input_shape=(TIME_STEPS, *input_shape)))
        else:
            time_distributed_model.add(TimeDistributed(model.layers[model_layer_index]))
    # Return
    return time_distributed_model


def _bn_relu(input, time_distributed=False, verbose=False):
    """Helper to build a BN -> relu block
    """
    if verbose:
        print("    _bn_relu")
    if time_distributed:
        norm = TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS))(input)
    else:
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    time_distributed = conv_params["time_distributed"]
    verbose = conv_params["verbose"]

    def f(input):
        if verbose:
            print("    _conv_bn_relu")
        if time_distributed:
            conv = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size,
                                          strides=strides, padding=padding,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer))(input)
        else:
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv, time_distributed, verbose)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    time_distributed = conv_params["time_distributed"]
    verbose = conv_params["verbose"]

    def f(input):
        if verbose:
            print("    _bn_relu_conv")
        activation = _bn_relu(input, time_distributed, verbose)
        if time_distributed:
            return TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size,
                                          strides=strides, padding=padding,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer))(activation)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual, time_distributed=False, verbose=False):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if verbose:
            print("    SHORTCUT : input -", shortcut.shape, "residual -", residual.shape)
            print("    SHORTCUT not right")
        if time_distributed:
            shortcut = TimeDistributed(Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                       kernel_size=(1, 1),
                                       strides=(stride_width, stride_height),
                                       padding="valid",
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=l2(0.0001)))(input)
        else:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

    if verbose:
        print("    SHORTCUT : input -", shortcut.shape, "residual -", residual.shape)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False, time_distributed=False, verbose=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):

            if verbose:
                print("  ", i, " residual block")

            init_strides = (1, 1)

            if i == 0 and not is_first_layer:
                init_strides = (2, 2)

            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0),
                                   time_distributed=time_distributed, verbose=verbose)(input)

        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, time_distributed=False, verbose=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if verbose:
            print("    basic block")

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            if time_distributed:
                conv1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3, 3),
                                               strides=init_strides,
                                               padding="same",
                                               kernel_initializer="he_normal",
                                               kernel_regularizer=l2(1e-4)))(input)
            else:
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=init_strides,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=init_strides,
                                  time_distributed=time_distributed, verbose=verbose)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                 time_distributed=time_distributed, verbose=verbose)(conv1)

        return _shortcut(input, residual, time_distributed, verbose)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, time_distributed=False, verbose=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if verbose:
            print("    bottleneck block")

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            if time_distributed:
                conv_1_1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(1, 1),
                                                  strides=init_strides,
                                                  padding="same",
                                                  kernel_initializer="he_normal",
                                                  kernel_regularizer=l2(1e-4)))(input)
            else:
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=init_strides,
                                  padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1), strides=init_strides,
                                     time_distributed=time_distributed, verbose=verbose)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                 time_distributed=time_distributed, verbose=verbose)(conv_1_1)

        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1),
                                 time_distributed=time_distributed, verbose=verbose)(conv_3_3)

        return _shortcut(input, residual, time_distributed, verbose)

    return f


def _handle_dim_ordering(time_distributed=False, verbose=False):
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if verbose:
        print("_handle_dim_ordering")
    if time_distributed:
        if K.image_dim_ordering() == 'tf':
            ROW_AXIS = 2
            COL_AXIS = 3
            CHANNEL_AXIS = -1
        else:
            CHANNEL_AXIS = 2
            ROW_AXIS = 3
            COL_AXIS = 4
    else:
        if K.image_dim_ordering() == 'tf':
            ROW_AXIS = 1
            COL_AXIS = 2
            CHANNEL_AXIS = 3
        else:
            CHANNEL_AXIS = 1
            ROW_AXIS = 2
            COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, time_distributed=False, verbose=False):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels), or
                         (nb_time_steps, nb_rows, nb_cols, nb_channels) in case of TimeDistributed
            num_outputs: The number of outputs at final softmax layer (at each time step)
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering(time_distributed, verbose)

        if time_distributed:
            if len(input_shape) != 4:
                raise Exception("Input shape should be a tuple (nb_time_steps, nb_rows, nb_cols, nb_channels)")
        else:
            if len(input_shape) != 3:
                raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")

        # Permute dimension order if necessary
        if time_distributed:
            if K.image_dim_ordering() == 'th':
                input_shape = (input_shape[4], input_shape[1], input_shape[2])
        else:
            if K.image_dim_ordering() == 'th':
                input_shape = (input_shape[2], input_shape[0], input_shape[1])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2),
                              time_distributed=time_distributed, verbose=verbose)(input)
        if time_distributed:
            pool1 = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))(conv1)
        else:
            pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        if verbose:
            print("Built input & 1st 7x7 layer")

        # Residual blocks
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            if verbose:
                print(i, "r =", r)
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0),
                                    time_distributed=time_distributed, verbose=verbose)(block)
            if verbose:
                print("Built residual block")
            filters *= 2

        # Last activation
        block = _bn_relu(block, time_distributed=time_distributed, verbose=verbose)
        if verbose:
            print("Built last activation block")

        # Classifier block
        block_shape = K.int_shape(block)
        if time_distributed:
            pool2 = TimeDistributed(AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                     strides=(1, 1)))(block)
            flatten1 = TimeDistributed(Flatten())(pool2)
            dense = TimeDistributed(Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax"))(flatten1)
        else:
            pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                     strides=(1, 1))(block)
            flatten1 = Flatten()(pool2)
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        if verbose:
            print("Built classifier block")

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, time_distributed=False, verbose=False):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2], time_distributed, verbose)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, time_distributed=False, verbose=False):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3], time_distributed, verbose)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, time_distributed=False, verbose=False):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3], time_distributed, verbose)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, time_distributed=False, verbose=False):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3], time_distributed, verbose)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, time_distributed=False, verbose=False):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3], time_distributed, verbose)

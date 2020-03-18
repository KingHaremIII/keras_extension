# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:43:41 2019

@author: kINGHAREM
"""
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.layers import Lambda, Layer
from keras.utils import conv_utils
from keras import activations, initializers, regularizers, constraints


class _ComplexConv(Layer):
    '''
    '''
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 output_merge=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
                (2, )+kernel_size, rank+1, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.output_merge = output_merge

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(input_shape)) + ' inputs.')

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        self.kernerl_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=self.kernerl_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(2, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        assert type(inputs) is list, 'must input a list containing two tensors'
        assert len(inputs) == 2, 'must input 2 inputs tensors, one for real'\
            + ', and the other for image.'

        if self.rank == 1:
            outputs_real = K.conv1d(
                inputs[0],
                self.kernel[0],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])\
                    - K.conv1d(
                inputs[1],
                self.kernel[1],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
            outputs_imag = K.conv1d(
                inputs[0],
                self.kernel[1],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])\
                + K.conv1d(
                inputs[1],
                self.kernel[0],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs_real = K.conv2d(
                inputs[0],
                self.kernel[0],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)\
                - K.conv2d(
                inputs[1],
                self.kernel[1],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
            outputs_imag = K.conv2d(
                inputs[0],
                self.kernel[1],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)\
                + K.conv2d(
                inputs[1],
                self.kernel[0],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs_real = K.conv3d(
                inputs[0],
                self.kernel[0],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)\
                - K.conv3d(
                inputs[1],
                self.kernel[1],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
            outputs_imag = K.conv3d(
                inputs[0],
                self.kernel[1],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)\
                + K.conv3d(
                inputs[1],
                self.kernel[0],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.output_merge:
            outputs = K.concatenate((K.expand_dims(outputs_real, axis=1),
                                     K.expand_dims(outputs_imag, axis=1)),
                                    axis=1)
            if self.use_bias:
                outputs = K.bias_add(
                    outputs,
                    self.bias,
                    data_format=self.data_format)

            if self.activation is not None:
                return self.activation(outputs)
            return outputs
        else:
            if self.use_bias:
                outputs_real = K.bias_add(
                    outputs_real,
                    self.bias[0],
                    data_format=self.data_format)
                outputs_imag = K.bias_add(
                    outputs_imag,
                    self.bias[1],
                    data_format=self.data_format)

            if self.activation is not None:
                return [self.activation(outputs_real),
                        self.activation(outputs_imag)]
            return [outputs_real, outputs_imag]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i+1],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            if self.output_merge:
                return (input_shape[0][0], 2) + tuple(new_space)\
                    + (self.filters,)
            else:
                out_shape_list = []
                out_shape_list.append(
                        (input_shape[0][0], ) + tuple(new_space)
                        + (self.filters,))
                out_shape_list.append(
                        (input_shape[0][0], ) + tuple(new_space)
                        + (self.filters,))
                return out_shape_list
        if self.data_format == 'channels_first':
            space = input_shape[0][2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i+1],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            if self.output_merge:
                return (input_shape[0][0], 2, self.filters) + tuple(new_space)
            else:
                out_shape_list = []
                out_shape_list.append(
                        (input_shape[0][0], self.filters) + tuple(new_space))
                out_shape_list.append(
                        (input_shape[0][0], self.filters) + tuple(new_space))
                return out_shape_list

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(
                    self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                    self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ComplexConv1D(_ComplexConv):
    '''1D complex convolution layer (e.g. temporal convolution).
    '''
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 output_merge=False,
                 **kwargs):
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError('When using causal padding in `Conv1D`, '
                                 '`data_format` must be "channels_last" '
                                 '(temporal data).')
        if isinstance(kernel_size, tuple):
            pass
        else:
            raise ValueError('kernel size of ComplexConv1D should be in the '
                             + 'format of tuple, detected'
                             + str(type(kernel_size)))
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            output_merge=output_merge,
            **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop('rank')
        return config


if __name__ == '__main__':
    key = input('test?(y/n)')
    if key == 'y':
        from keras import Input, Model
        input_real = Input((3, 1), name='channellastReal1D')
        input_imag = Input((3, 1), name='channellastImag1D')
        CConv1d = ComplexConv1D(
                5, kernel_size=(3, ), use_bias=False,
                output_merge=False,
                name='ComplexConv1D')
        output_tensor = CConv1d([input_real, input_imag])
        model = Model([input_real, input_imag], output_tensor,
                      name='TestModel')
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())

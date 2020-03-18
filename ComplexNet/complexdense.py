# -*- coding: utf-8 -*-

# Abstract: complex dense layer, which can deal with complex computation,
#    with input of 2 channels(one for real and the other for image part), and
#    output of channel-separated format or merged format.(by specify <output_merge>)
# Authors: Zuyao Hong
# ==================================
# Derive from Chiheb Trabelsi's work
# ==================================
#
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer


class ComplexDenseNet():
    '''Construct a network composed llinearly of specific layers of complex
           dense, additionally, you can command the net to use
           LeakyReLU activation(by 'LReLU', 'LeakyReLU' or 'leakyrelu') function.
    '''
    def __init__(self, No_, layers, units,
                 output_merge=False,
                 activation=None,
                 leakyrelu_alpha=0.3,
                 use_bias=True,
                 init_criterion='he',    # 'glorot'
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if type(units) is list:
            if len(units) == layers:
                self.units_list = units
            else:
                raise ValueError('If you want to set units of complex dense'
                                 + ' layers respectively, the length of units'
                                 + ' list should be according to the value of'
                                 + ' <layers>. \nGot '+str(len(units))+' units'
                                 + 'values and the value of <layers> is '
                                 + str(layers) + ', however.')
        elif type(units) is int:
            self.units_list = [units for i in range(layers)]
        else:
            raise ValueError('Irregular input of <units>, which should be'
                             + 'a list or int.')
        self.No_ = No_
        self.layers = layers
        self.output_merge = output_merge
        self.activation = activation
        if activation in {'LReLU', 'LeakyReLU', 'leakyrelu'}:
            self.leakyrelu_alpha = leakyrelu_alpha
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.seed = seed
        self.kwargs = kwargs

    def __call__(self, inputs):
        if type(inputs) is list:
            if len(inputs) != 2:
                raise ValueError('input should be one tensor or'
                                 + 'a list of tensor containing two'
                                 + ' tensor. \nGot '+str(len(inputs))
                                 + ' tensors, however')
            else:
                self.input_real = inputs[0]
                self.input_imag = inputs[1]
        else:
            raise ValueError('you must input a list containing two tensor'
                             + '<error position>: ComplexNet'+str(self.No_))
        try:
            self.units_list
        except AttributeError:
            raise ValueError('This class has no attribute of units/'
                             + 'units list, which is necessary to build'
                             + 'the complex value network.')

        if self.layers > 1:
            [output_real, output_imag] = ComplexDense(
                        units=self.units_list[0],
                        activation=self.activation,
                        leakyrelu_alpha=self.leakyrelu_alpha,
                        use_bias=self.use_bias,
                        init_criterion=self.init_criterion,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        seed=self.seed,
                        output_merge=False,
                        name=self.No_+'CDense1',
                        **self.kwargs)([self.input_real, self.input_imag])

            if self.layers > 2:
                for i in range(2, self.layers):
                    [output_real, output_imag] = ComplexDense(
                            units=self.units_list[i-1],
                            activation=self.activation,
                            leakyrelu_alpha=self.leakyrelu_alpha,
                            use_bias=self.use_bias,
                            init_criterion=self.init_criterion,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            activity_regularizer=self.activity_regularizer,
                            kernel_constraint=self.kernel_constraint,
                            bias_constraint=self.bias_constraint,
                            seed=self.seed,
                            output_merge=False,
                            name=self.No_+'CDense'+str(i),
                            **self.kwargs)([output_real, output_imag])

            if self.output_merge:
                return ComplexDense(
                        units=self.units_list[-1],
                        activation=self.activation,
                        leakyrelu_alpha=self.leakyrelu_alpha,
                        use_bias=self.use_bias,
                        init_criterion=self.init_criterion,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        seed=self.seed,
                        output_merge=True,
                        name=self.No_+'CDense'+str(self.layers),
                        **self.kwargs)([output_real, output_imag])
            else:
                return ComplexDense(
                        units=self.units_list[-1],
                        activation=self.activation,
                        leakyrelu_alpha=self.leakyrelu_alpha,
                        use_bias=self.use_bias,
                        init_criterion=self.init_criterion,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        seed=self.seed,
                        output_merge=False,
                        name=self.No_+'CDense'+str(self.layers),
                        **self.kwargs)([output_real, output_imag])

        else:
            return ComplexDense(
                        units=self.units_list[0],
                        activation=self.activation,
                        leakyrelu_alpha=self.leakyrelu_alpha,
                        use_bias=self.use_bias,
                        init_criterion=self.init_criterion,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        activity_regularizer=self.activity_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        seed=self.seed,
                        output_merge=self.output_merge,
                        name=self.No_+'CDense',
                        **self.kwargs)([self.input_real, self.input_imag])


class ComplexDense(Layer):

    def __init__(self, units,
                 activation=None,   # support LeakyReLU by specify
                                    # 'LReLU' or 'LeakyReLU' or 'leakyrelu'
                 leakyrelu_alpha=0.3,
                 use_bias=True,
                 init_criterion='he',    # 'glorot'
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 output_merge=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        if activation in {'LReLU', 'LeakyReLU', 'leakyrelu'}:
            self.activation = 'LReLU'
            self.leakyrelu_alpha = leakyrelu_alpha
        else:
            self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.output_merge = output_merge
        # self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        # assert len(input_shape) == 2
        # assert input_shape[-1] % 2 == 0
        # input_dim = input_shape[-1] // 2
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(input_shape)) + ' inputs.')

        input_dim = input_shape[0][-1]
        self.input_dim = input_dim
        data_format = K.image_data_format()
        kernel_shape = (input_dim, 2*self.units)
        fan_in, fan_out = initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )
        fan_in = tf.to_float(fan_in)
        fan_out = tf.to_float(fan_out)

        if self.init_criterion == 'he':
            s = K.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = K.sqrt(1. / (fan_in + fan_out))

        # rng = RandomStreams(seed=self.seed)

        # Equivalent initialization using amplitude phase representation:
        """modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        def init_w_real(shape, dtype=None):
            return modulus * K.cos(phase)
        def init_w_imag(shape, dtype=None):
            return modulus * K.sin(phase)"""

        # Initialization using euclidean representation:
        def init_w(shape, dtype=None):

            return tf.random_normal(shape=kernel_shape,
                                    mean=0.0,
                                    stddev=s,
                                    dtype=tf.float32,
                                    seed=self.seed,
                                    name=None
                                    )
            # return rng.normal(
            #     size=kernel_shape,
            #     avg=0,
            #     std=s,
            #     dtype=dtype
            # )

        if self.kernel_initializer in {'complex'}:
            weight_init = init_w
        else:
            weight_init = self.kernel_initializer

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=weight_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        # self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        # input_shape = K.shape(inputs)
        # input_dim = input_shape[-1] // 2
        # real_input = inputs[:, :input_dim]
        # imag_input = inputs[:, input_dim:]
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')
        real_input = inputs[0]
        imag_input = inputs[1]

        inputs = K.concatenate([real_input, imag_input], axis=-1)
        # print(inputs.shape)
        # print(self.real_kernel.shape)
        # print(self.imag_kernel.shape)
        cat_kernels_4_real = K.concatenate(
            [self.kernel[:, 0:self.units], self.kernel[:, self.units:]],
            axis=-1
        )

        cat_kernels_4_imag = K.concatenate(
            [-self.kernel[:, self.units:], self.kernel[:, 0:self.units]],
            axis=-1
        )
        cat_kernels_4_complex = K.concatenate(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = K.dot(inputs, cat_kernels_4_complex)
        # print(output.shape)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            if self.activation == 'LReLU':
                output = K.relu(output, alpha=self.leakyrelu_alpha)
            else:
                output = self.activation(output)

        if self.output_merge:
            output_real = K.transpose(output)[0:self.units]
            output_imag = K.transpose(output)[self.units:]
            output_real = K.expand_dims(K.transpose(output_real), 1)
            output_imag = K.expand_dims(K.transpose(output_imag), 1)
            return K.concatenate((output_real, output_imag), axis=1)
        else:
            output_real = K.transpose(K.transpose(output)[0:self.units])
            output_imag = K.transpose(K.transpose(output)[self.units:])
            return [output_real, output_imag]

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        # assert input_shape[-1]
        # output_shape = list(input_shape)
        # output_shape[-1] = 2 * self.units
        output_shape = list(input_shape[0])
        if self.output_merge:
            output_shape.insert(1, 2)
            output_shape[-1] = self.units
            return tuple(output_shape)
        else:
            output_shape[-1] = self.units
            return [tuple(output_shape), tuple(output_shape)]

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                    self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(
                    self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        if self.activation == 'LReLU':
            config['activation'] = self.activation
        else:
            config['activation'] = activations.serialize(self.activation)
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    key = input('press 0 for testing <ComplexDenseNet> and press 1 for '
                + 'testing <ComplexDense>, and the others for skipping.\n')
    if key == '0':
        from keras import Input, Model
        input_real = Input((3, ), name='input_real')
        input_imag = Input((3, ), name='input_image')
        output_tensor = ComplexDenseNet(
                No_='test', layers=3, units=[2, 3, 5],
                output_merge=True, activation='leakyrelu')([input_real,
                                                            input_imag])
        model = Model([input_real, input_imag], output_tensor,
                      name='TestModel')
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())
    if key == '1':
        from keras import Input, Model
        input_real = Input((3, ), name='input_real')
        input_imag = Input((3, ), name='input_image')
        [output_r, output_i] = ComplexDense(
                units=2, output_merge=False,
                name='TestCDense1', activation='leakyrelu')([input_real,
                                                             input_imag])
        [output_r, output_i] = ComplexDense(
                units=3, output_merge=False,
                name='TestCDense2', activation='leakyrelu')([output_r,
                                                             output_i])
        output_tensor = ComplexDense(
                units=5, output_merge=True,
                name='TestCDense3', activation='leakyrelu')([output_r,
                                                             output_i])
        model = Model([input_real, input_imag], output_tensor,
                      name='TestModel')
        model.compile(optimizer='adam', loss='mse')
        print('Merged Version: \n')
        print(model.summary())

        [output_r, output_i] = ComplexDense(
                units=5, output_merge=False,
                name='TestCDense', activation='leakyrelu')([input_real,
                                                            input_imag])
        model = Model([input_real, input_imag], [output_r, output_i],
                      name='TestModel')
        model.compile(optimizer='adam', loss='mse')
        print('Not Merged Version: \n')
        print(model.summary())

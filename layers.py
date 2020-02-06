import math
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as k
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec


class NoisyDense(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 sigma_init=0.017,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 std_func=None,
                 sigma_func=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(NoisyDense, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.sigma_init = sigma_init
        self.std_func = std_func
        self.sigma_func = sigma_func
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

        # pylint was complaining
        self.mu_weights = None
        self.sigma_weights = None
        self.mu_bias = None
        self.sigma_bias = None

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or k.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `NoisyDense` layer with non-floating point'
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `NoisyDense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        if self.std_func is None:
            std = math.sqrt(3 / input_shape[-1])
        else:
            std = self.std_func(input_shape[-1])
        if self.sigma_func is not None:
            sigma_init = self.sigma_func(self.sigma_init, input_shape[-1])
        else:
            sigma_init = self.sigma_init
        self.mu_weights = self.add_weight(
            'mu_weights',
            shape=[last_dim, self.units],
            initializer=initializers.RandomUniform(minval=-std, maxval=std),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.sigma_weights = self.add_weight(
            'sigma_weights',
            shape=[last_dim, self.units],
            initializer=initializers.Constant(value=sigma_init),
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.mu_bias = self.add_weight(
                'mu_bias',
                shape=[self.units, ],
                initializer=initializers.RandomUniform(minval=-std, maxval=std),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            self.sigma_bias = self.add_weight(
                'sigma_bias',
                shape=[self.units, ],
                initializer=initializers.Constant(value=sigma_init),
                dtype=self.dtype,
                trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        rank = len(inputs.shape)
        epsilon_w = tf.random.normal(self.sigma_weights.shape)
        w = self.mu_weights + tf.multiply(self.sigma_weights, epsilon_w)
        if rank > 2:
            outputs = tf.tensordot(inputs, w, [[rank - 1], [0]])
        else:
            inputs = tf.cast(inputs, self._compute_dtype)
            outputs = tf.matmul(inputs, w)
        if self.use_bias:
            epsilon_b = tf.random.normal(self.sigma_bias.shape)
            b = self.mu_bias + tf.multiply(self.sigma_bias, epsilon_b)
            outputs = tf.nn.bias_add(outputs, b)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


class FactorizedNoisyDense(NoisyDense):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 sigma_init=0.5,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 std_func=None,
                 sigma_func=None,
                 **kwargs):
        if std_func is None:
            def std_func(p):
                return -1 / math.sqrt(p)
        if sigma_func is None:
            def sigma_func(sigma, p):
                return sigma / math.sqrt(p)
        super(FactorizedNoisyDense, self).__init__(units,
                                                   activation=activation,
                                                   use_bias=use_bias,
                                                   sigma_init=sigma_init,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   kernel_constraint=kernel_constraint,
                                                   bias_constraint=bias_constraint,
                                                   std_func=std_func,
                                                   sigma_func=sigma_func,
                                                   **kwargs)

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        rank = len(inputs.shape)
        func = lambda x: tf.sign(x) * tf.sqrt(tf.abs(x))
        epsilon_i = func(tf.random.normal([inputs.shape[-1], 1]))
        epsilon_j = func(tf.random.normal([1, self.units]))
        epsilon_w = tf.matmul(epsilon_i, epsilon_j)
        w = self.mu_weights + tf.multiply(self.sigma_weights, epsilon_w)
        if rank > 2:
            outputs = tf.tensordot(inputs, w, [[rank - 1], [0]])
        else:
            inputs = tf.cast(inputs, self._compute_dtype)
            outputs = tf.matmul(inputs, w)
        if self.use_bias:
            b = self.mu_bias + tf.multiply(self.sigma_bias, tf.squeeze(epsilon_j, 0))
            outputs = tf.nn.bias_add(outputs, b)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

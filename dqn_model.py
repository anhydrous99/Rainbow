import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from layers import NoisyDense, FactorizedNoisyDense

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def dense_chooser(dense=None):
    if dense == 'noisy':
        return NoisyDense
    elif dense == 'factorized_noisy':
        return FactorizedNoisyDense
    else:
        return keras.layers.Dense


class DQN(tf.Module):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None, dueling=False):
        super(DQN, self).__init__(name=name)
        dense = dense_chooser(use_dense)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation='relu')(inp)
        x = keras.layers.Conv2D(64, 4, 2, activation='relu')(x)
        x = keras.layers.Conv2D(64, 3, 1, activation='relu')(x)
        x = keras.layers.Flatten()(x)

        x1 = dense(256 if dueling else 512, activation='relu')(x)
        x1 = dense(n_actions)(x1)

        if dueling:
            x2 = dense(256, activation='relu')(x)
            x2 = dense(1)(x2)

            x_mean = tf.keras.layers.Lambda(lambda xs: tf.math.reduce_mean(xs, axis=1, keepdims=True))(x1)
            x = tf.keras.layers.Subtract()([x1, x_mean])
            x = tf.keras.layers.Add()([x2, x])
        self.model = Model(inputs=inp, outputs=x if dueling else x1)

    @tf.function
    def __call__(self, x):
        return self.model(x)


class DQNNoConvolution(tf.Module):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None, dueling=False):
        super(DQNNoConvolution, self).__init__(name=name)
        dense = dense_chooser(use_dense)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = None

        x1 = dense(sum(input_shape) * 6, activation='tanh')(inp)
        x1 = dense(n_actions)(x1)

        if dueling:
            x2 = dense(sum(input_shape) * 6, activation='tanh')(inp)
            x2 = dense(1)(x2)

            x_mean = tf.keras.layers.Lambda(lambda xs: tf.math.reduce_mean(xs, axis=1, keepdims=True))(x1)
            x = tf.keras.layers.Subtract()([x1, x_mean])
            x = tf.keras.layers.Add()([x2, x])
        self.model = Model(inputs=inp, outputs=x if dueling else x1)
        self.model.summary()

    @tf.function
    def __call__(self, x):
        return self.model(x)

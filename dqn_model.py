import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
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
    def __init__(self, input_shape, n_actions, name=None, use_dense=None):
        super(DQN, self).__init__(name=name)
        dense = dense_chooser(use_dense)
        self.model = Sequential([
            keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation='relu'),
            keras.layers.Conv2D(64, 4, 2, activation='relu'),
            keras.layers.Conv2D(64, 3, 1, activation='relu'),
            keras.layers.Flatten(),
            dense(512, activation='relu'),
            dense(n_actions)
        ])

    @tf.function
    def __call__(self, x):
        return self.model(x)


class DQNNoConvolution(tf.Module):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None):
        super(DQNNoConvolution, self).__init__(name=name)
        dense = dense_chooser(use_dense)
        self.model = Sequential([
            dense(sum(input_shape) * 6, input_shape=input_shape, activation='tanh'),
            dense(n_actions)
        ])

    @tf.function
    def __call__(self, x):
        return self.model(x)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from layers import NoisyDense, FactorizedNoisyDense

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) != 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device[0], True)


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

        advantage = dense(256 if dueling else 512, activation='relu')(x)
        advantage = dense(n_actions)(advantage)

        if dueling:
            value = dense(256, activation='relu')(x)
            value = dense(1)(value)

            advantage_m = tf.keras.layers.Lambda(lambda xs: tf.math.reduce_mean(xs, axis=1, keepdims=True))(advantage)
            x = tf.keras.layers.Subtract()([advantage, advantage_m])
            x = tf.keras.layers.Add()([value, x])
        else:
            x = advantage
        self.model = Model(inputs=inp, outputs=x)

    @tf.function
    def __call__(self, x):
        return self.model(x)


class DQNNoConvolution(tf.Module):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None, dueling=False):
        super(DQNNoConvolution, self).__init__(name=name)
        dense = dense_chooser(use_dense)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = dense(sum(input_shape), activation='relu')(inp)
        advantage = dense(sum(input_shape) * 6, activation='tanh')(x)
        advantage = dense(n_actions)(advantage)

        if dueling:
            value = dense(sum(input_shape) * 6, activation='tanh')(x)
            value = dense(1)(value)

            advantage_m = tf.keras.layers.Lambda(lambda xs: tf.math.reduce_mean(xs, axis=1, keepdims=True))(advantage)
            x = tf.keras.layers.Subtract()([advantage, advantage_m])
            x = tf.keras.layers.Add()([value, x])
        else:
            x = advantage
        self.model = Model(inputs=inp, outputs=x)
        self.model.summary()

    @tf.function
    def __call__(self, x):
        return self.model(x)

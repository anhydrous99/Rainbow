import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from layers import NoisyDense, FactorizedNoisyDense, DuelingAggregator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) != 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


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

        if dueling:
            adv = dense(512, activation='relu')(x)
            adv = dense(n_actions)(adv)

            val = dense(512, activation='relu')(x)
            val = dense(1)(val)

            x = DuelingAggregator()([adv, val])
        else:
            x = dense(512, activation='relu')(x)
            x = dense(n_actions)(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.summary()

    @tf.function
    def __call__(self, x):
        return self.model(x)


class DQNNoConvolution(tf.Module):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None, dueling=False):
        super(DQNNoConvolution, self).__init__(name=name)
        dense = dense_chooser(use_dense)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = dense(50, activation='relu')(inp)

        if dueling:
            adv = dense(25, activation='relu')(x)
            adv = dense(25, activation='relu')(adv)
            adv = dense(n_actions)(adv)

            val = dense(25, activation='relu')(x)
            val = dense(25, activation='relu')(val)
            val = dense(1)(val)

            x = DuelingAggregator()([adv, val])
        else:
            x = dense(50, activation='relu')(x)
            x = dense(50, activation='relu')(x)
            x = dense(n_actions)(x)

        self.model = Model(inputs=inp, outputs=x)
        self.model.summary()

    @tf.function
    def __call__(self, x):
        return self.model(x)

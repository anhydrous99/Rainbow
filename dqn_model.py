import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DQN(tf.Module):
    def __init__(self, input_shape, n_actions, name=None):
        super(DQN, self).__init__(name=name)

        self.model = Sequential([
            keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation='relu'),
            keras.layers.Conv2D(64, 4, 2, activation='relu'),
            keras.layers.Conv2D(64, 3, 1, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(n_actions)
        ])

    @tf.function
    def __call__(self, x):
        return self.model(x)

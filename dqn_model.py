import tensorflow as tf


class DQN(tf.Module):
    def __init__(self, input_shape, n_actions, name=None):
        super(DQN, self).__init__(name=name)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation='relu'),
            tf.keras.layers.Conv2D(64, 4, 2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(n_actions, activation='relu')
        ])

    def __call__(self, x):
        return self.model(x)

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


class DQNBase(tf.Module):
    def __init__(self, name=None, use_distributional=False, n_atoms=None, v_min=None, v_max=None):
        super(DQNBase, self).__init__(name=name)
        self.use_dist = use_distributional
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        if use_distributional:
            assert n_atoms is not None
            assert v_min is not None
            assert v_max is not None
            delta_z = (v_max - v_min) / n_atoms
            self.supports = tf.range(v_min, v_max, delta_z, name='supports')  # z_i in the paper

    def _reshape_output_tensor(self, x):
        batch_size = x.shape[0]
        return tf.reshape(x, [batch_size, -1, self.n_atoms])

    @tf.function
    def q_values(self, x):
        net_output = self._reshape_output_tensor(self.model(x))
        probabilities = tf.nn.softmax(net_output, axis=-1)
        weights = probabilities * self.supports
        return tf.reduce_sum(weights, axis=-1)


class DQN(DQNBase):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None, dueling=False, use_distributional=False,
                 n_atoms=None, v_min=None, v_max=None):
        super(DQN, self).__init__(name=name, use_distributional=use_distributional, n_atoms=n_atoms,
                                  v_min=v_min, v_max=v_max)
        dense = dense_chooser(use_dense)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = keras.layers.Conv2D(32, 8, 4, input_shape=input_shape, activation='relu')(inp)
        x = keras.layers.Conv2D(64, 4, 2, activation='relu')(x)
        x = keras.layers.Conv2D(64, 3, 1, activation='relu')(x)
        x = keras.layers.Flatten()(x)

        if dueling:
            adv = dense(512, activation='relu')(x)
            adv = dense(n_actions * n_atoms if use_distributional else n_actions)(adv)

            val = dense(512, activation='relu')(x)
            val = dense(n_atoms if use_distributional else 1)(val)

            x = DuelingAggregator(n_atoms)([adv, val])
        else:
            x = dense(512, activation='relu')(x)
            x = dense(n_actions * n_atoms if use_distributional else n_actions)(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.summary()

    @tf.function
    def __call__(self, x):
        net_output = self.model(x)
        if self.use_dist:
            net_output = self._reshape_output_tensor(net_output)
        return net_output


class DQNNoConvolution(DQNBase):
    def __init__(self, input_shape, n_actions, name=None, use_dense=None, dueling=False, use_distributional=False,
                 n_atoms=None, v_min=None, v_max=None):
        super(DQNNoConvolution, self).__init__(name=name, use_distributional=use_distributional, n_atoms=n_atoms,
                                               v_min=v_min, v_max=v_max)
        dense = dense_chooser(use_dense)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = dense(50, activation='relu')(inp)

        if dueling:
            adv = dense(25, activation='relu')(x)
            adv = dense(25, activation='relu')(adv)
            adv = dense(n_actions * n_atoms if use_distributional else n_actions)(adv)

            val = dense(25, activation='relu')(x)
            val = dense(25, activation='relu')(val)
            val = dense(n_atoms if use_distributional else 1)(val)

            x = DuelingAggregator(n_atoms)([adv, val])
        else:
            x = dense(50, activation='relu')(x)
            x = dense(50, activation='relu')(x)
            x = dense(n_actions * n_atoms if use_distributional else n_actions)(x)

        self.model = Model(inputs=inp, outputs=x)
        self.model.summary()

    @tf.function
    def __call__(self, x):
        net_output = self.model(x)
        if self.use_dist:
            net_output = self._reshape_output_tensor(net_output)
        return net_output

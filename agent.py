import tensorflow as tf
import experience as xp
import numpy as np
import dqn_model
import os


def compute_projection(next_distribution, rewards, dones, v_min, v_max, n_atoms, gamma):
    delta_z = (v_max - v_min) / (n_atoms - 1)
    atoms = tf.tile(tf.expand_dims(tf.range(n_atoms, dtype=tf.float32), axis=0), [32, 1])
    rewards = tf.tile(tf.expand_dims(rewards, axis=-1), [1, 51])
    m = tf.zeros_like(next_distribution)
    tz = np.minimum(v_max, tf.maximum(v_min, rewards + (v_min + atoms * delta_z) * gamma))
    b = (tz - v_min) / delta_z
    l = tf.math.floor(b)
    u = tf.math.ceil(b)
    l_int = tf.cast(l, dtype=tf.int32)
    u_int = tf.cast(u, dtype=tf.int32)
    l_update = next_distribution * (u - b)
    u_update = next_distribution * (b - l)
    ml = tf.gather_nd(l_update, l_int, batch_dims=1)
    mu = tf.gather_nd(u_update, u_int, batch_dims=1)
    print(m)


class Agent:
    def __init__(self, env, replay_size, optimizer, batch_size, n_steps, gamma, use_double=True, use_dense=None,
                 dueling=False, use_distributional=False, n_atoms=None, v_min=None, v_max=None):
        net = dqn_model.DQN if len(env.observation_space.shape) != 1 else dqn_model.DQNNoConvolution
        self.env = env
        self.state = None
        self.total_reward = 0.0
        self.n_steps = n_steps
        self.use_double = use_double
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.exp_buffer = xp.PriorityBuffer(replay_size, gamma, n_steps)
        self.net = net(env.observation_space.shape, env.action_space.n, use_dense=use_dense, dueling=dueling,
                       use_distributional=use_distributional, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.tgt_net = net(env.observation_space.shape, env.action_space.n, use_dense=use_dense, dueling=dueling,
                           use_distributional=use_distributional, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.params = self.net.trainable_variables
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_distributional = use_distributional
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = tf.convert_to_tensor(state_a)
            if self.use_distributional:
                q_vals_v = self.net.q_values(state_v)
            else:
                q_vals_v = self.net(state_v)
            act_v = tf.math.argmax(q_vals_v, axis=1)
            action = int(act_v.numpy()[0])

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = xp.Experience(self.state, action, reward, is_done)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def sync_weights(self):
        self.tgt_net.model.set_weights(self.net.model.get_weights())

    def load_checkpoint(self, path):
        if os.path.exists(path):
            print('Loading checkpoint')
            self.net.model.load_weights(path)
            self.tgt_net.model.set_weights(self.net.model.get_weights())

    def save_checkpoint(self, path):
        self.net.model.save_weights(path)

    # @tf.function
    # def ll(self, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights):
    #     if self.use_double:
    #         states_t = tf.concat([states_t, next_states_t], 0)
    #     net_output = self.net(states_t)
    #     state_action_values = tf.squeeze(
    #         tf.gather(net_output[:self.batch_size], tf.expand_dims(actions_t, 1), batch_dims=1), -1)
    #     if self.use_double:
    #         next_state_actions = tf.argmax(net_output[self.batch_size:], axis=1)
    #         next_state_values = tf.squeeze(
    #             tf.gather(self.tgt_net(next_states_t), tf.expand_dims(next_state_actions, 1), batch_dims=1), -1)
    #     else:
    #         next_state_values = tf.reduce_max(self.tgt_net(next_states_t), axis=1)
    #     state_action_values = tf.where(done_mask, tf.zeros_like(next_state_values), state_action_values)
    #     next_state_values = tf.stop_gradient(next_state_values)
    #
    #     # Bellman equation
    #     expected_state_action_values = next_state_values * (gamma ** self.n_steps) + rewards_t
    #
    #     # Calculate loss
    #     losses = tf.math.squared_difference(expected_state_action_values, state_action_values)
    #     losses = tf.math.multiply(weights, losses)
    #     return tf.reduce_mean(losses, axis=-1), tf.add(1.0e-5, losses)

    def ll(self, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights):
        if self.use_double:
            states_t = tf.concat([states_t, next_states_t], 0)
        net_output = self.net(states_t)

        # Calculate the state action values
        state_action_values = tf.squeeze(tf.gather(net_output[:self.batch_size], tf.reshape(actions_t, [-1, 1, 1]),
                                                   batch_dims=1))
        state_action_values = tf.nn.log_softmax(state_action_values, axis=-1)

        # Calculate next state distribution
        next_distribution_t, next_q_values_t = self.tgt_net.both(next_states_t)
        next_actions = tf.argmax(next_q_values_t, axis=-1)
        next_distribution = tf.nn.softmax(next_distribution_t, axis=1)
        next_best_distribution = tf.squeeze(tf.gather(next_distribution, tf.reshape(next_actions, [-1, 1, 1]),
                                                      batch_dims=1))

        # Calculate projection
        projected_distribution = compute_projection(next_best_distribution, rewards_t, done_mask, self.v_min,
                                                    self.v_max, self.n_atoms, gamma)

    def calc_loss(self, batch, gamma):
        states, actions, rewards, dones, next_states, weights = batch[:6]

        states_t = tf.convert_to_tensor(states)
        next_states_t = tf.convert_to_tensor(next_states)
        actions_t = tf.convert_to_tensor(actions)
        rewards_t = tf.convert_to_tensor(rewards)
        done_mask = tf.convert_to_tensor(dones, dtype=bool)
        weights_t = tf.convert_to_tensor(weights)
        return self.ll(gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights_t)

    def step(self, gamma, update_exp_weights=True):
        batch = self.exp_buffer.sample(self.batch_size)
        indices = batch[6]
        with tf.GradientTape() as tape:
            loss_t, losses = self.calc_loss(batch, gamma)
        gradient = tape.gradient(loss_t, self.params)
        self.optimizer.apply_gradients(zip(gradient, self.params))
        if update_exp_weights:
            self.exp_buffer.update_weights(indices, losses.numpy())

    def buffer_size(self):
        return len(self.exp_buffer)

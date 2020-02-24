import tensorflow as tf
import experience as xp
import numpy as np
import dqn_model
import os


class Agent:
    def __init__(self, env, replay_size, optimizer, batch_size, n_steps, gamma, use_double=True, use_dense=None,
                 dueling=False, use_categorical=False, n_atoms=None, v_min=None, v_max=None, use_priority=False,
                 alpha=0.6, beta=0.4, train_steps=5000000):
        net = dqn_model.DQN if len(env.observation_space.shape) != 1 else dqn_model.DQNNoConvolution
        self.env = env
        self.state = None
        self.update_count = 0
        self.total_reward = 0.0
        self.n_steps = n_steps
        self.use_double = use_double
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.beta = beta
        self.use_priority = use_priority
        if use_priority:
            self.exp_buffer = xp.PriorityBuffer(replay_size, gamma, n_steps, alpha)
        else:
            self.exp_buffer = xp.ExperienceBuffer(replay_size, gamma, n_steps)
        self.net = net(env.observation_space.shape, env.action_space.n, use_dense=use_dense, dueling=dueling,
                       use_distributional=use_categorical, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.tgt_net = net(env.observation_space.shape, env.action_space.n, use_dense=use_dense, dueling=dueling,
                           use_distributional=use_categorical, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.params = self.net.trainable_variables
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_categorical = use_categorical
        self.train_steps = train_steps
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
            if self.use_categorical:
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

    @tf.function
    def ll(self, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights):
        if self.use_double:
            states_t = tf.concat([states_t, next_states_t], 0)
        net_output = self.net(states_t)

        # Calculate the current state action values
        state_action_values = tf.squeeze(
            tf.gather(net_output[:self.batch_size], tf.expand_dims(actions_t, 1), batch_dims=1), -1)
        state_action_values = tf.where(done_mask, tf.zeros_like(state_action_values), state_action_values)

        # Calculate the next state action values
        if self.use_double:
            next_state_actions = tf.argmax(net_output[self.batch_size:], axis=1)
            next_state_values = tf.squeeze(
                tf.gather(self.tgt_net(next_states_t), tf.expand_dims(next_state_actions, 1), batch_dims=1), -1)
        else:
            next_state_values = tf.reduce_max(self.tgt_net(next_states_t), axis=1)
        next_state_values = tf.stop_gradient(next_state_values)

        # Bellman equation
        expected_state_action_values = next_state_values * (gamma ** self.n_steps) + rewards_t

        # Calculate loss
        losses = tf.math.squared_difference(expected_state_action_values, state_action_values)
        losses = tf.math.multiply(weights, losses)
        return tf.reduce_mean(losses, axis=-1), tf.add(1.0e-5, losses)

    @tf.function
    def ll_dist(self, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights):
        if self.use_double:
            states_t = tf.concat([states_t, next_states_t], 0)
        # Calculate current state probabilities
        net_output = self.net(states_t)
        state_action_dist = tf.nn.log_softmax(net_output[:self.batch_size], axis=-1)
        state_action_dist = tf.squeeze(tf.gather(state_action_dist, tf.reshape(actions_t, [-1, 1, 1]), batch_dims=1))

        # Calculate next state probabilities
        target_net_output = tf.nn.softmax(self.tgt_net(next_states_t), axis=-1)
        if self.use_double:
            next_state_actions = tf.nn.softmax(net_output[self.batch_size:])
            next_best_actions = tf.argmax(tf.reduce_sum(next_state_actions, -1), -1)
        else:
            next_best_actions = tf.argmax(tf.reduce_sum(target_net_output, -1), -1)
        next_state_dist = tf.squeeze(tf.gather(target_net_output, tf.reshape(next_best_actions, [-1, 1, 1]),
                                               batch_dims=1))

        # Calculate the Bellman operator T to produce Tz
        delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        support = tf.linspace(self.v_min, self.v_max, self.n_atoms)
        Tz = tf.expand_dims(rewards_t, -1) + tf.expand_dims(tf.cast(tf.logical_not(done_mask), tf.float32), -1) * (
                gamma ** self.n_steps) * tf.expand_dims(support, 0)
        Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = tf.math.floor(b)
        u = tf.math.floor(b)

        # Fix disappearing probability mass
        eq_mask = tf.equal(l, u)
        u_greater = tf.greater(u, 0)
        l_less = tf.less(l, self.n_atoms - 1.0)
        l = tf.where(tf.logical_and(eq_mask, u_greater), x=l - 1, y=l)
        u = tf.where(tf.logical_and(eq_mask, l_less), x=u + 1, y=u)

        m = tf.zeros(self.batch_size * self.n_atoms)
        offset = tf.linspace(0.0, ((self.batch_size - 1.0) * self.n_atoms), self.batch_size)
        offset = tf.reshape(tf.tile(tf.expand_dims(offset, -1), [1, self.n_atoms]), [-1, 1])
        m = tf.tensor_scatter_nd_add(
            m,
            tf.cast(tf.reshape(l, [-1, 1]) + offset, tf.int32),
            tf.reshape(next_state_dist * (u - b), [-1])
        )
        m = tf.tensor_scatter_nd_add(
            m,
            tf.cast(tf.reshape(u, [-1, 1]) + offset, tf.int32),
            tf.reshape(next_state_dist * (b - l), [-1])
        )
        m = tf.reshape(m, [self.batch_size, self.n_atoms])

        # Calculate loss
        losses = -tf.reduce_sum(m * state_action_dist, -1)
        losses = tf.multiply(weights, losses)
        return tf.reduce_mean(losses, -1), losses

    def calc_loss(self, batch, gamma):
        if self.use_priority:
            states, actions, rewards, dones, next_states, weights = batch[:6]
            weights_t = tf.convert_to_tensor(weights)
        else:
            states, actions, rewards, dones, next_states = batch[:5]
            weights_t = tf.ones_like(rewards)

        states_t = tf.convert_to_tensor(states)
        next_states_t = tf.convert_to_tensor(next_states)
        actions_t = tf.convert_to_tensor(actions)
        rewards_t = tf.convert_to_tensor(rewards)
        done_mask = tf.convert_to_tensor(dones, dtype=bool)
        if self.use_categorical:
            return self.ll_dist(gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights_t)
        return self.ll(gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, weights_t)

    def step(self, gamma, update_exp_weights=True):
        indices = None
        beta = min(1.0, self.update_count / (self.train_steps * (1.0 - self.beta)) + self.beta)
        batch = self.exp_buffer.sample(self.batch_size, beta)
        if self.use_priority:
            indices = batch[6]
        with tf.GradientTape() as tape:
            loss_t, losses = self.calc_loss(batch, gamma)
        gradient = tape.gradient(loss_t, self.params)
        self.optimizer.apply_gradients(zip(gradient, self.params))
        if self.use_priority and update_exp_weights:
            self.exp_buffer.update_weights(indices, losses.numpy())
        self.update_count += 1

    def buffer_size(self):
        return len(self.exp_buffer)

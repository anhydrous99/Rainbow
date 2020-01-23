import atari_wrappers
import dqn_model

import time
import numpy as np
import collections

import tensorflow as tf

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), \
               np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.total_reward = 0.0
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = tf.convert_to_tensor(state_a)
            q_vals_v = net(state_v)
            _, act_v = tf.math.reduce_max(q_vals_v, axis=1)
            action = int(act_v.numpy())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch

    states_t = tf.convert_to_tensor(states)
    next_states_t = tf.convert_to_tensor(next_states)
    actions_t = tf.convert_to_tensor(actions)
    rewards_t = tf.convert_to_tensor(rewards)
    done_mask = tf.convert_to_tensor(dones, dtype=tf.uint8)

    state_action_values = net(states_t)


def train(env_name='PongNoFrameskip-v4',
          mean_reward_bound=19.5,
          gamma=0.99,
          batch_size=32,
          replay_size=10000,
          replay_start_size=10000,
          learning_rate=1.0e-4,
          sync_target_frames=1000,
          epsilon_decay_last_frame=10**5,
          epsilon_start=1.0,
          epsilon_final=0.02):
    env = atari_wrappers.make_env(env_name)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    print(net.model)

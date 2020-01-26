import atari_wrappers
import dqn_model

import os
import time
import numpy as np
import collections

import tensorflow as tf
from numpy_ringbuffer import RingBuffer
from tensorflow.python.eager import profiler

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

        def ar(array, **kwargs):
            return np.array(array, **kwargs)

        return ar(states), ar(actions), ar(rewards, dtype=np.float32), ar(dones, dtype=np.uint8), ar(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.state = None
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
            act_v = tf.math.argmax(q_vals_v, axis=1)
            action = int(act_v.numpy()[0])

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


@tf.function
def ll(net, tgt_net, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask):
    net_output = net(states_t)
    net_output = tf.gather(net_output, tf.expand_dims(actions_t, 1), batch_dims=1)
    state_action_values = tf.squeeze(net_output, -1)
    next_state_values = tf.math.reduce_max(tgt_net(next_states_t), axis=1)
    state_action_values = tf.where(done_mask, tf.zeros_like(next_state_values), state_action_values)
    next_state_values = tf.stop_gradient(next_state_values)

    expected_state_action_values = next_state_values * gamma + rewards_t
    return tf.keras.losses.MSE(expected_state_action_values, state_action_values)


def calc_loss(batch, net, tgt_net, gamma, tape):
    states, actions, rewards, dones, next_states = batch

    states_t = tf.convert_to_tensor(states)
    next_states_t = tf.convert_to_tensor(next_states)
    actions_t = tf.convert_to_tensor(actions)
    rewards_t = tf.convert_to_tensor(rewards)
    done_mask = tf.convert_to_tensor(dones, dtype=bool)
    tape.watch(states_t)
    return ll(net, tgt_net, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask)


def train(env_name='PongNoFrameskip-v4',
          mean_reward_bound=19.5,
          gamma=0.99,
          batch_size=32,
          replay_size=1000000,
          replay_start_size=50000,
          learning_rate=0.00025,
          sync_target_frames=10000,
          epsilon_decay_last_frame=1000000,
          gradient_momentum=0.95,
          squared_gradient_momentum=0.95,
          min_squared_gradient=0.01,
          epsilon_start=1.0,
          epsilon_final=0.1,
          train_frames=50000000):
    profiler.start_profiler_server(6009)
    print(f'Training DQN on {env_name} environment')
    env = atari_wrappers.make_env(env_name)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.model.summary()

    if os.path.exists(f'checkpoints/{env_name}/checkpoint'):
        print('Loading checkpoint')
        net.model.load_weights(f'checkpoints/{env_name}/checkpoint')
        tgt_net.model.set_weights(net.model.get_weights())

    buffer = ExperienceBuffer(replay_size)
    agent = Agent(env, buffer)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                            momentum=gradient_momentum,
                                            rho=squared_gradient_momentum,
                                            epsilon=min_squared_gradient)
    params = net.trainable_variables
    total_rewards = RingBuffer(capacity=100000)
    rewards_mean_std = []
    frame_idx = 0
    count = 0
    update_count = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay_last_frame)

        reward = agent.play_step(net, epsilon)
        if reward is not None:
            count += 1
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print(f'{frame_idx}: done {count} games, mean reward: {mean_reward}, eps {epsilon}, speed: {speed}')
            if best_mean_reward is None or best_mean_reward < mean_reward:
                # Save network
                net.model.save_weights(f'./checkpoints/{env_name}/checkpoint')
                if best_mean_reward is not None:
                    print(f'Best mean reward updated {best_mean_reward} -> {mean_reward}, model saved')
                best_mean_reward = mean_reward
            if frame_idx > train_frames:
                print(f'Trained for {frame_idx} frames. Done.')
                break

        if len(buffer) < replay_start_size:
            continue

        if frame_idx % sync_target_frames == 0:
            tgt_net.model.set_weights(net.model.get_weights())

        batch = buffer.sample(batch_size)
        with tf.GradientTape() as tape:
            loss_t = calc_loss(batch, net, tgt_net, gamma, tape)
        gradient = tape.gradient(loss_t, params)
        optimizer.apply_gradients(zip(gradient, params))
        update_count += 1
        if update_count % 10000 == 0:
            arr = np.array(total_rewards[-10000:])
            rewards_mean_std.append({'rewards_mean': np.mean(arr),
                                     'rewards_std': np.std(arr)})
    return rewards_mean_std

import os
import plot
import time
import wrappers
import dqn_model
import numpy as np
import experience as xp
import tensorflow as tf


class Agent:
    def __init__(self, env, replay_size, optimizer, batch_size, n_steps, gamma):
        net = dqn_model.DQN if len(env.observation_space.shape) != 1 else dqn_model.DQNNoConvolution
        self.env = env
        self.state = None
        self.total_reward = 0.0
        self.exp_buffer = xp.ExperienceBuffer(replay_size, gamma, n_steps)
        self.net = net(env.observation_space.shape, env.action_space.n)
        self.tgt_net = net(env.observation_space.shape, env.action_space.n)
        self.optimizer = optimizer
        self.batch_size = batch_size
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
    def ll(self, gamma, states_t, next_states_t, actions_t, rewards_t, done_mask, n_steps=3):
        net_output = self.net(states_t)
        net_output = tf.gather(net_output, tf.expand_dims(actions_t, 1), batch_dims=1)
        state_action_values = tf.squeeze(net_output, -1)
        next_state_values = tf.math.reduce_max(self.tgt_net(next_states_t), axis=1)
        state_action_values = tf.where(done_mask, tf.zeros_like(next_state_values), state_action_values)
        next_state_values = tf.stop_gradient(next_state_values)

        expected_state_action_values = next_state_values * (gamma ** n_steps) + rewards_t
        return tf.keras.losses.MSE(expected_state_action_values, state_action_values)

    def calc_loss(self, batch, gamma):
        states, actions, rewards, dones, next_states = batch

        states_t = tf.convert_to_tensor(states)
        next_states_t = tf.convert_to_tensor(next_states)
        actions_t = tf.convert_to_tensor(actions)
        rewards_t = tf.convert_to_tensor(rewards)
        done_mask = tf.convert_to_tensor(dones, dtype=bool)
        return self.ll(gamma, states_t, next_states_t, actions_t, rewards_t, done_mask)

    def step(self, gamma):
        params = self.net.trainable_variables
        batch = self.exp_buffer.sample(self.batch_size)
        with tf.GradientTape() as tape:
            loss_t = self.calc_loss(batch, gamma)
        gradient = tape.gradient(loss_t, params)
        self.optimizer.apply_gradients(zip(gradient, params))

    def buffer_size(self):
        return len(self.exp_buffer)


def train(env_name='PongNoFrameskip-v4',
          gamma=0.99,
          batch_size=32,
          replay_size=1000000,
          replay_start_size=50000,
          learning_rate=0.00025,
          sync_target_frames=10000,
          epsilon_decay_last_frame=1000000,
          epsilon_start=1.0,
          epsilon_final=0.1,
          train_frames=50000000,
          train_rewards=495,
          n_steps=3,
          save_checkpoints=True):
    print(f'Training DQN on {env_name} environment')
    env = wrappers.make_env(env_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    agent = Agent(env, replay_size, optimizer, batch_size, n_steps, gamma)
    if save_checkpoints:
        agent.load_checkpoint(f'checkpoints/{env_name}/checkpoint')

    total_rewards = []
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

        reward = agent.play_step(epsilon)
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
                if save_checkpoints:
                    agent.save_checkpoint(f'./checkpoints/{env_name}/checkpoint')
                if best_mean_reward is not None:
                    print(f'Best mean reward updated {best_mean_reward} -> {mean_reward}, model saved')
                best_mean_reward = mean_reward
            if train_frames is not None:
                if frame_idx >= train_frames:
                    print(f'Trained for {frame_idx} frames. Done.')
                    break
            if train_rewards is not None:
                if mean_reward >= train_rewards:
                    print(f'Reached reward: {mean_reward}. Done.')
                    break

        if agent.buffer_size() < replay_start_size:
            continue

        if frame_idx % sync_target_frames == 0:
            agent.sync_weights()
        agent.step(gamma)
        update_count += 1
        if update_count % 100 == 0:
            arr = np.array(total_rewards[-100:])
            rewards_mean_std.append({'rewards_mean': np.mean(arr),
                                     'rewards_std': np.std(arr),
                                     'step': update_count})
    plot.directory_check('./plots')
    plot.plot(rewards_mean_std, f'./plots/{env_name}_{str(int(time.time()))}.png', env_name)
    plot.directory_check('./data')
    plot.save(rewards_mean_std, f'./data/{env_name}_{str(int(time.time()))}.csv')

import plot
import time
import wrappers
import numpy as np
import tensorflow as tf
from agent import Agent


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
          save_checkpoints=True,
          run_name=None,
          use_double=True,
          use_dense=None,
          dueling=False,
          priority_replay=None,
          categorical=None,
          record=False,
          random_seed=None,
          index=0):
    n_atoms = v_min = v_max = None
    use_categorical = False
    if categorical is not None:
        use_categorical = True
        n_atoms = categorical['n_atoms']
        v_min = categorical['v'][0]
        v_max = categorical['v'][1]

    alpha = beta = None
    use_priority_replay = False
    if priority_replay is not None:
        use_priority_replay = True
        alpha = priority_replay['alpha']
        beta = priority_replay['beta']

    print(f'Training DQN on {env_name} environment')
    print(f'Params: gamma:{gamma}, batch_size:{batch_size}, replay_size:{replay_size}')
    print(f'        replay_start_size: {replay_start_size}, learning_rate:{learning_rate}')
    print(f'        sync_target_frames: {sync_target_frames}, epsilon_decay_last_frame:{epsilon_decay_last_frame}')
    print(f'        epsilon_start: {epsilon_start}, epsilon_final: {epsilon_final}, train_frames: {train_frames}')
    print(f'        train_rewards: {train_rewards}, n_steps: {n_steps}, save_checkpoints: {save_checkpoints}')
    print(f'        run_name: {run_name}, use_double: {use_double}, use_dense: {use_dense}, dueling: {dueling}')
    if use_categorical:
        print(f'        categorical - n_atoms: {n_atoms}, v_min: {v_min}, v_max: {v_max}')
    if use_priority_replay:
        print(f'        priority buffer - alpha: {alpha} beta: {beta}')
    print(f'        random_seed: {random_seed}, index: {index}')
    f_name = env_name + "_" + run_name if run_name is not None else env_name
    env = wrappers.make_env(env_name, record, f_name)
    if random_seed is not None:
        tf.random.set_seed(random_seed)
        env.seed(random_seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1.5e-4)
    agent = Agent(env, replay_size, optimizer, batch_size, n_steps, gamma, use_double, use_dense, dueling,
                  use_categorical, n_atoms, v_min, v_max, train_frames if train_frames is not None else 5000000)
    if save_checkpoints:
        agent.load_checkpoint(f'checkpoints/{f_name}/checkpoint')

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
            print(f'{index}:{frame_idx}: done {count} games, mean reward: {mean_reward}, eps {epsilon}, speed: {speed}')
            if best_mean_reward is None or best_mean_reward < mean_reward:
                # Save network
                if best_mean_reward is not None:
                    if save_checkpoints:
                        agent.save_checkpoint(f'./checkpoints/{f_name}/checkpoint')
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
        agent.step(gamma, True if update_count % 1000 == 0 else False)
        update_count += 1
        rewards_mean_std.append({'reward': total_rewards[-1:][0],
                                 'step': update_count})
    env.close()
    plot.directory_check('./plots')
    plot.plot(rewards_mean_std, f'./plots/{f_name}.png', f_name)
    plot.directory_check('./data')
    plot.save(rewards_mean_std, f'./data/{f_name}.csv')

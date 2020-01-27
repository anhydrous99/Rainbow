import argparse
from train import train


def main():
    parser = argparse.ArgumentParser(description='Deep Q-network (DQN)')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--sync_target_frames', type=int, default=5000)
    parser.add_argument('--replay_start_size', type=int, default=10000)
    parser.add_argument('--epsilon_decay_last_frame', type=int, default=100000)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_final', type=float, default=0.1)
    parser.add_argument('--train_frames', type=int, default=5000000)
    parser.add_argument('--train_reward', type=int, default=475)
    args = parser.parse_args()
    env_str = args.env
    gamma = args.gamma
    batch_size = args.batch_size
    replay_size = args.replay_size
    learning_rate = args.learning_rate
    sync_target_frames = args.sync_target_frames
    replay_start_size = args.replay_start_size
    epsilon_decay_last_frame = args.epsilon_decay_last_frame
    epsilon_start = args.epsilon_start
    epsilon_final = args.epsilon_final
    train_frames = args.train_frames
    train_reward = args.train_reward

    train(env_str,
          gamma,
          batch_size,
          replay_size,
          replay_start_size,
          learning_rate,
          sync_target_frames,
          epsilon_decay_last_frame,
          epsilon_start,
          epsilon_final,
          train_frames,
          train_reward)


if __name__ == '__main__':
    main()

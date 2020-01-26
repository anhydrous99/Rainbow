import argparse
from train import train


def main():
    parser = argparse.ArgumentParser(description='Deep Q-network (DQN)')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--mean_reward_bound', type=float, default=19.5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--sync_target_frames', type=int, default=10000)
    parser.add_argument('--replay_start_size', type=int, default=50000)
    parser.add_argument('--epsilon_decay_last_frame', type=int, default=1000000)
    parser.add_argument('--gradient_momentum', type=float, default=0.95)
    parser.add_argument('--squared_gradient_momentum', type=float, default=0.95)
    parser.add_argument('--min_squared_gradient', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_final', type=float, default=0.1)
    parser.add_argument('--train_frames', type=int, default=50000000)
    args = parser.parse_args()
    env_str = args.env
    mean_reward_bound = args.mean_reward_bound
    gamma = args.gamma
    batch_size = args.batch_size
    replay_size = args.replay_size
    learning_rate = args.learning_rate
    sync_target_frames = args.sync_target_frames
    replay_start_size = args.replay_start_size
    epsilon_decay_last_frame = args.epsilon_decay_last_frame
    gradient_momentum = args.gradient_momentum
    squared_gradient_momentum = args.squared_gradient_momentum
    min_squared_gradient = args.min_squared_gradient
    epsilon_start = args.epsilon_start
    epsilon_final = args.epsilon_final
    train_frames = args.train_frames

    train(env_str,
          mean_reward_bound,
          gamma,
          batch_size,
          replay_size,
          replay_start_size,
          learning_rate,
          sync_target_frames,
          epsilon_decay_last_frame,
          gradient_momentum,
          squared_gradient_momentum,
          min_squared_gradient,
          epsilon_start,
          epsilon_final,
          train_frames)


if __name__ == '__main__':
    main()

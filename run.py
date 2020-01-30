import json
import argparse
from train import train


def main():


    parser = argparse.ArgumentParser(description='Deep Q-network (DQN)')
    parser.add_argument('--conf', type=str, default='conf.json')
    args = parser.parse_args()
    file = open(args.conf, mode='r')
    conf_text = file.read()
    file.close()
    conf_json = json.loads(conf_text)

    gamma = conf_json['gamma']
    batch_size = conf_json['batch_size']
    replay_size = conf_json['replay_size']
    learning_rate = conf_json['learning_rate']
    sync_target_frames = conf_json['sync_target_frames']
    replay_start_size = conf_json['replay_start_size']
    epsilon_decay_last_frame = conf_json['epsilon_decay_last_frame']
    epsilon_start = conf_json['epsilon_start']
    epsilon_final = conf_json['epsilon_final']

    runs = conf_json['runs']

    for run in runs:
        env_str = run['env']

        ngamma = run['gamma'] if 'gamma' in run else gamma
        nbatch_size = run['batch_size'] if 'batch_size' in run else batch_size
        nreplay_size = run['replay_size'] if 'replay_size' in run else replay_size
        nlearning_rate = run['learning_rate'] if 'learning_rate' in run else learning_rate
        nsync_target_frames = run['sync_target_frames'] if 'sync_target_frames' in run else sync_target_frames
        nreplay_start_size = run['replay_start_size'] if 'replay_start_size' in run else replay_start_size
        nepsilon_decay_last_frame = run['epsilon_decay_last_frame'] if 'epsilon_decay_last_frame' in run else epsilon_decay_last_frame
        nepsilon_start = run['epsilon_start'] if 'epsilon_start' in run else epsilon_start
        nepsilon_final = run['epsilon_final'] if 'epsilon_final' in run else epsilon_final

        if 'train_reward' in run:
            train_reward = run['train_reward']
            train(env_str,
                  ngamma,
                  nbatch_size,
                  nreplay_size,
                  nreplay_start_size,
                  nlearning_rate,
                  nsync_target_frames,
                  nepsilon_decay_last_frame,
                  nepsilon_start,
                  nepsilon_final,
                  None,
                  train_reward)
        elif 'train_frames' in run:
            train_frames = run['train_frames']
            train(env_str,
                  ngamma,
                  nbatch_size,
                  nreplay_size,
                  nreplay_start_size,
                  nlearning_rate,
                  nsync_target_frames,
                  nepsilon_decay_last_frame,
                  nepsilon_start,
                  nepsilon_final,
                  train_frames,
                  None)


if __name__ == '__main__':
    main()

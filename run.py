import ray
import json
import argparse
from train import train
from utils import conditional_decorator


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

    if conf_json['multiprocessing']:
        ray.init()

    @conditional_decorator(ray.remote(num_cpus=conf_json['num_cpus'], num_gpus=conf_json['num_gpus']),
                           conf_json['multiprocessing'])
    def fun(f_run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
            epsilon_decay_last_frame, epsilon_start, epsilon_final):
        env_str = f_run['env']
        ngamma = f_run['gamma'] if 'gamma' in f_run else gamma
        nbatch_size = f_run['batch_size'] if 'batch_size' in f_run else batch_size
        nreplay_size = f_run['replay_size'] if 'replay_size' in f_run else replay_size
        nlearning_rate = f_run['learning_rate'] if 'learning_rate' in f_run else learning_rate
        nsync_target_frames = f_run['sync_target_frames'] if 'sync_target_frames' in f_run else sync_target_frames
        nreplay_start_size = f_run['replay_start_size'] if 'replay_start_size' in f_run else replay_start_size
        nepsilon_decay_last_frame = f_run[
            'epsilon_decay_last_frame'] if 'epsilon_decay_last_frame' in f_run else epsilon_decay_last_frame
        nepsilon_start = f_run['epsilon_start'] if 'epsilon_start' in f_run else epsilon_start
        nepsilon_final = f_run['epsilon_final'] if 'epsilon_final' in f_run else epsilon_final
        if 'train_reward' in f_run:
            train_reward = f_run['train_reward']
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
        elif 'train_frames' in f_run:
            train_frames = f_run['train_frames']
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
        return 1

    remote_objects = []
    for run in runs:
        if conf_json['multiprocessing']:
            robj = fun.remote(run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                              epsilon_decay_last_frame, epsilon_start, epsilon_final)
            remote_objects.append(robj)
        else:
            fun(run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                epsilon_decay_last_frame, epsilon_start, epsilon_final)
    if conf_json['multiprocessing']:
        values = ray.get(remote_objects)
    for val in values:
        assert val == 1


if __name__ == '__main__':
    main()

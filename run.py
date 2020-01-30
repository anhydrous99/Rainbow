import ray
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
    ray.init()

    @ray.remote(num_cpus=2, num_gpus=0.2)
    def fun(frun, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
            epsilon_decay_last_frame, epsilon_start, epsilon_final):
        env_str = frun['env']
        ngamma = frun['gamma'] if 'gamma' in frun else gamma
        nbatch_size = frun['batch_size'] if 'batch_size' in frun else batch_size
        nreplay_size = frun['replay_size'] if 'replay_size' in frun else replay_size
        nlearning_rate = frun['learning_rate'] if 'learning_rate' in frun else learning_rate
        nsync_target_frames = frun['sync_target_frames'] if 'sync_target_frames' in frun else sync_target_frames
        nreplay_start_size = frun['replay_start_size'] if 'replay_start_size' in frun else replay_start_size
        nepsilon_decay_last_frame = frun[
            'epsilon_decay_last_frame'] if 'epsilon_decay_last_frame' in frun else epsilon_decay_last_frame
        nepsilon_start = frun['epsilon_start'] if 'epsilon_start' in frun else epsilon_start
        nepsilon_final = frun['epsilon_final'] if 'epsilon_final' in frun else epsilon_final
        if 'train_reward' in frun:
            train_reward = frun['train_reward']
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
        elif 'train_frames' in frun:
            train_frames = frun['train_frames']
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
        robj = fun.remote(run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                          epsilon_decay_last_frame, epsilon_start, epsilon_final)
        remote_objects.append(robj)
    values = ray.get(remote_objects)
    for val in values:
        assert val == 1


if __name__ == '__main__':
    main()

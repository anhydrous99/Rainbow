import json
import argparse
from train import train
from joblib import Parallel, delayed


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
    n_steps = conf_json['n_steps']
    save_checkpoints = conf_json['save_checkpoints']
    use_double = conf_json['use_double']
    use_dense = conf_json['use_dense'] if 'use_dense' in conf_json else None
    random_seed = conf_json['random_seed'] if 'random_seed' in conf_json else None
    dueling = conf_json['dueling'] if 'dueling' in conf_json else False
    runs = conf_json['runs']

    def fun(s_args):
        f_run, f_gamma, f_batch_size, f_replay_size, f_learning_rate, f_sync_target_frames = s_args[:6]
        f_replay_start_size, f_e_d_l_frame, f_epsilon_start, f_epsilon_final, f_n_steps = s_args[6:11]
        f_save_checkpoints, f_use_double, f_use_dense, f_dueling, f_random_seed = s_args[11:]
        env_str = f_run['env']
        n_gamma = f_run['gamma'] if 'gamma' in f_run else f_gamma
        n_batch_size = f_run['batch_size'] if 'batch_size' in f_run else f_batch_size
        n_replay_size = f_run['replay_size'] if 'replay_size' in f_run else f_replay_size
        n_learning_rate = f_run['learning_rate'] if 'learning_rate' in f_run else f_learning_rate
        n_sync_target_frames = f_run['sync_target_frames'] if 'sync_target_frames' in f_run else f_sync_target_frames
        n_replay_start_size = f_run['replay_start_size'] if 'replay_start_size' in f_run else f_replay_start_size
        n_epsilon_decay_last_frame = f_run[
            'epsilon_decay_last_frame'] if 'epsilon_decay_last_frame' in f_run else f_e_d_l_frame
        n_epsilon_start = f_run['epsilon_start'] if 'epsilon_start' in f_run else f_epsilon_start
        n_epsilon_final = f_run['epsilon_final'] if 'epsilon_final' in f_run else f_epsilon_final
        nn_steps = f_run['n_steps'] if 'n_steps' in f_run else f_n_steps
        run_name = f_run['run_name'] if 'run_name' in f_run else None
        n_use_double = f_run['use_double'] if 'use_double' in f_run else f_use_double
        n_use_dense = f_run['use_dense'] if 'use_dense' in f_run else f_use_dense
        n_dueling = f_run['dueling'] if 'dueling' in f_run else f_dueling
        train_frames = None
        train_reward = None
        if 'train_frames' in f_run:
            train_frames = f_run['train_frames']
        if 'train_reward' in f_run:
            train_reward = f_run['train_reward']
        train(env_str,
              n_gamma,
              n_batch_size,
              n_replay_size,
              n_replay_start_size,
              n_learning_rate,
              n_sync_target_frames,
              n_epsilon_decay_last_frame,
              n_epsilon_start,
              n_epsilon_final,
              train_frames,
              train_reward,
              nn_steps,
              f_save_checkpoints,
              run_name,
              n_use_double,
              n_use_dense,
              n_dueling,
              f_random_seed)
        return 1

    if 'multiprocessing' in conf_json and conf_json['multiprocessing']:
        concurrent_processes = conf_json['concurrent_processes']
        args = []
        for run in runs:
            args.append((run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                         epsilon_decay_last_frame, epsilon_start, epsilon_final, n_steps, save_checkpoints, use_double,
                         use_dense, dueling, random_seed))
        Parallel(n_jobs=concurrent_processes, prefer='processes')(delayed(fun)(arg) for arg in args)
    else:
        for run in runs:
            args = (run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                    epsilon_decay_last_frame, epsilon_start, epsilon_final, n_steps, save_checkpoints, use_double,
                    use_dense, dueling, random_seed)
            fun(args)


if __name__ == '__main__':
    main()

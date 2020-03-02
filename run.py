import json
import argparse
from train import train
from joblib import Parallel, delayed


def json_checker(ret_obj, json_obj, field):
    return json_obj[field] if field in json_obj else ret_obj


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
    use_dense = json_checker(None, conf_json, 'use_dense')
    random_seed = json_checker(None, conf_json, 'random_seed')
    dueling = json_checker(False, conf_json, 'use_dueling')
    priority_replay = json_checker(None, conf_json, 'priority_replay')
    categorical = json_checker(None, conf_json, 'categorical')
    record = json_checker(False, conf_json, 'record_training')
    runs = conf_json['runs']

    def fun(s_args):
        (f_run, f_gamma, f_batch_size, f_replay_size, f_learning_rate, f_sync_target_frames, f_replay_start_size,
         f_epsilon_decay_last_frame, f_epsilon_start, f_epsilon_final, f_n_steps, f_save_checkpoints, f_use_double,
         f_use_dense, f_dueling, f_priority_replay, f_categorical, f_record, f_random_seed, f_index) = s_args
        env_str = f_run['env']
        n_gamma = json_checker(f_gamma, f_run, 'gamma')
        n_batch_size = json_checker(f_batch_size, f_run, 'batch_size')
        n_replay_size = json_checker(f_replay_size, f_run, 'replay_size')
        n_learning_rate = json_checker(f_learning_rate, f_run, 'learning_rate')
        n_sync_target_frames = json_checker(f_sync_target_frames, f_run, 'sync_target_frames')
        n_replay_start_size = json_checker(f_replay_start_size, f_run, 'replay_start_size')
        n_epsilon_decay_last_frame = json_checker(f_epsilon_decay_last_frame, f_run, 'epsilon_decay_last_frame')
        n_epsilon_start = json_checker(f_epsilon_start, f_run, 'epsilon_start')
        n_epsilon_final = json_checker(f_epsilon_final, f_run, 'epsilon_final')
        nn_steps = json_checker(f_n_steps, f_run, 'n_steps')
        run_name = json_checker(None, f_run, 'run_name')
        n_use_double = json_checker(f_use_double, f_run, 'use_double')
        n_use_dense = json_checker(f_use_dense, f_run, 'use_dense')
        n_dueling = json_checker(f_dueling, f_run, 'use_dueling')
        n_priority_replay = json_checker(f_priority_replay, f_run, 'priority_replay')
        n_categorical = json_checker(f_categorical, f_run, 'categorical')
        n_record = json_checker(f_record, f_run, 'record_training')
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
              n_priority_replay,
              n_categorical,
              n_record,
              f_random_seed,
              f_index)
        return 1

    if 'multiprocessing' in conf_json and conf_json['multiprocessing']:
        concurrent_processes = conf_json['concurrent_processes']
        args = []
        for index, run in enumerate(runs):
            args.append((run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                         epsilon_decay_last_frame, epsilon_start, epsilon_final, n_steps, save_checkpoints, use_double,
                         use_dense, dueling, priority_replay, categorical, record, random_seed, index))
        Parallel(n_jobs=concurrent_processes, prefer='processes')(delayed(fun)(arg) for arg in args)
    else:
        for index, run in enumerate(runs):
            args = (run, gamma, batch_size, replay_size, learning_rate, sync_target_frames, replay_start_size,
                    epsilon_decay_last_frame, epsilon_start, epsilon_final, n_steps, save_checkpoints, use_double,
                    use_dense, dueling, priority_replay, categorical, record, random_seed, index)
            fun(args)


if __name__ == '__main__':
    main()

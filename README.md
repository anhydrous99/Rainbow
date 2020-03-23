# Rainbow
This project is a Tensorflow 2.0 implementation of Rainbow (https://arxiv.org/abs/1710.02298).

After the creation of DQN in 2013 (https://arxiv.org/abs/1312.5602) and there after the improvements in 2015 (https://arxiv.org/abs/1312.5602)
the peeps in DeepMind pooled together 7 improvements to the DQN from 2015.

The improvements include
* Double Q-learning - (https://arxiv.org/abs/1509.06461)
* Prioritized Experience Replay - (https://arxiv.org/abs/1511.05952)
* Dueling network - (https://arxiv.org/abs/1511.06581)
* Multi-step learning - (https://link.springer.com/article/10.1007/BF00115009)
* Distributional RL - (https://arxiv.org/abs/1707.06887)
* Noisy Nets - (https://arxiv.org/abs/1706.10295)

My reason for creating this project really is to learn about Reinforcement learning and to use it as a base for future research into DQN with the goal to understand these concepts and, at the same time, create a more capable Agent.

Unfortunately, I don't have the resources to test all 47 Atari games tested in the Rainbow paper over the 200 million frames.
Still, I will try to get similar results in a couple of games then compare with the results, of that games, with the Rainbow paper.

## Requirements 

* Python greater than or equal to 3.7
* Tensorflow greater than or equal to 2.0
* matplotlib
* pandas
* seaborn
* gym
* atari-py
* OpenCV
* Cython
* NumPy
* GCC/VSCompiler
* Joblib

## Setup

I tend to use conda to setup my environment, be sure to have either gcc (for linux & mac) or VC (for windows):
```
conda create --name Rainbow python=3.7 tensorflow-gpu pandas matplotlib seaborn opencv cython joblib
conda activate Rainbow
pip install gym atari-py
```
To build the Cython code:
```
python setup.py build_ext --inplace
```
After setting up your run settings in `conf.json` to run:
```
python run.py
```
If you would like to record, you do need to install some more packages:
```
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

## Comparison

Here will the future comparison between the Rainbow paper and this project's results.

## Conguration

The hyper-parameters, the environment, the run settings, and the run specific hyper-parameters are set through the 
`conf.json` file with the possibility to have it named differently by using the `--conf <path to configuration file>` ticker 
option. The possible tunable hyper-parameters can be tuned globally, in the outer scope of the JSON object or locally, 
within the scope of the run JSON object.

### DQN hyper-parameters

`conf.json` configurable hyper-parameters:
* `multiprocessing` - Whether to have multiple runs at one via python's multiprocessing module.
* `concurrent_processes` - The number of concurrent processes to run when the multiprocessing ticker is true.
* `gamma` - A DQN hyper-parameters for how long in the past to take into account.
* `batch_size` - The batch size, while training.
* `replay_size` - The size of the replay buffer.
* `learning_rate` - The Adam algorithm's learning rate.
* `sync_target_frames` - The number of frames between when the target and regular network's weight's are synced.
* `replay_start_size` - The is the number of frames to fill the buffer with before starting the training processes.
* `epsilon_decay_last_frame` - The frame number when `epsilon_final` is reached for the epsilon-greedy algorithm.
* `epsilon_start` - The epsilon to start when using the epsilon-greedy algorithm to explore.
* `epsilon_final` - The epsilon to stop when using the epsilon-greedy algorithm.
* `save_checkpoints` - Whether to save checkpoints of the model as training is progressing.
* `random_seed` - The random seed to use with Tensorflow, NumPy, and OpenAI's gym.
* `run` - an array run JSON objects containing `env` - The, OpenAI gym, environment, `run_name` the name of the run, 
and `train_frames` and/or `train_reward` the reward to frame to stop training at. Most hyper-parameters described in
the global scope of our JSON file can be overwritten here, on a per run basis.
* `record_training` - A true or false where true indicates to record all episodes when training.

### N-Step DQN

* `n_steps` - The number of steps to un-role the Bellman equation.

### Double DQN

* `use_double` - Whether to use Double-DQN when training.

### Dueling DQN

* `use_dueling` - Whether to use Dueling DQN when training.

### Noisy layers

* `use_dense` - takes a string with three possible values, "factorized_noisy", "noisy", or false/null. If 
"factorized_noisy" the factorized random algorithm will be used, if "noisy" regular random algorithm will be used 
otherwise a regular `Dense` layer will be used.

### Priority Replay

* `priority_replay` - If this field exists then it is assumed that a priority replay buffer is to be used. This field would
be a JSON object with the fields `alpha` and `beta` which are priority replay hyper-parameters.

### Categorical DQN

* `categorical` - If this field exists then it is assumed that Categorical DQN will be used. This field would be a JSON 
object with the fields `n_atoms`, the number of atoms in the distributions, and `v` which is an array with two floats that
represent Vmin and Vmax in the Distributional RL paper.

## Output run metrics and checkpoints

If the `save_checkpoints` is true then the model is saved anytime a new best reward is reached. It will be saved in a folder
called checkpoints and a sub-folder whose name is the concatenation of the environment name and the run name. After a run 
has completed, in a data folder and a sub-folder names as with the checkpoints, a CSV file with the reward of the run
every step is saved. A plot of the reward is also saved, in a folder named plot.

# Show Cases
## Space Invaders
[![Space Invaders as played by Rainbow](https://img.youtube.com/v1/w6j5TxZZaIg/0.jpg)](https://www.youtube.com/watch?v=w6j5TxZZaIg)
## Pong
[![Pong as played by Rainbow](https://img.youtube.com/v1/zP6QJoJAyIs/0.jpg)](https://www.youtube.com/watch?v=zP6QJoJAyIs)

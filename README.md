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

## Comparison

Here will the future comparison between the Rainbow paper and this project's results.

import collections
import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done'])


def ar(array, **kwargs):
    return np.array(array, **kwargs)


class ExperienceBuffer:
    def __init__(self, capacity, gamma=0.99, n_steps=3):
        self.buffer = collections.deque(maxlen=capacity)
        self.gamma = gamma
        self.n_steps = n_steps
        self.capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer) - self.n_steps, batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for index in indices:
            current_buffer = self.buffer[index]
            current_state = current_buffer[0]
            current_action = current_buffer[1]
            current_done = current_buffer[3]
            reward = current_buffer[2]
            next_state = self.buffer[index + self.n_steps][0]
            for sub_index in range(1, self.n_steps):
                reward += self.buffer[index + sub_index][2] * (self.gamma ** sub_index)
                if self.buffer[index + sub_index][3]:
                    break
            states.append(current_state)
            actions.append(current_action)
            rewards.append(reward)
            dones.append(current_done)
            next_states.append(next_state)

        return ar(states), ar(actions), ar(rewards, dtype=np.float32), ar(dones, dtype=np.uint8), ar(next_states)


class PriorityBuffer:
    def __init__(self, capacity, gamma=0.99, n_steps=2, alpha=0.6):
        self.buffer = []
        self.position = 0
        self.alpha = alpha
        self.gamma = gamma
        self.n_steps = n_steps
        self.capacity = capacity
        it_cap = 1
        while it_cap < capacity:
            it_cap *= 2
        self._it_sum = SumSegmentTree(it_cap)
        self._it_min = MinSegmentTree(it_cap)
        self._max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        position = self.position
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[position] = experience
        self.position = (position + 1) % self.capacity
        self._it_sum[position] = self._max_priority ** self.alpha
        self._it_min[position] = self._max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        total = self._it_sum.sum(0, len(self.buffer) - (1 + self.n_steps))
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefix_sum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0.4):
        assert beta > 0
        indices = self._sample_proportional(batch_size)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for index in indices:
            current_buffer = self.buffer[index]
            current_state = current_buffer[0]
            current_action = current_buffer[1]
            current_done = current_buffer[3]
            reward = current_buffer[2]
            next_state = self.buffer[index + self.n_steps][0]
            for sub_index in range(1, self.n_steps):
                reward += self.buffer[index + sub_index][2] * (self.gamma ** sub_index)
                if self.buffer[index + sub_index][3]:
                    break
            states.append(current_state)
            actions.append(current_action)
            rewards.append(reward)
            dones.append(current_done)
            next_states.append(next_state)

        sm = self._it_sum.sum()
        p_min = self._it_min.min() / sm
        max_weight = (p_min * len(self.buffer)) ** (-beta)
        p_sample = self._it_sum[indices] / sm
        weights = (p_sample * len(self.buffer)) ** (-beta) / max_weight
        return ar(states), ar(actions), ar(rewards, dtype=np.float32), ar(dones, dtype=np.uint8), ar(next_states), \
               ar(weights, dtype=np.float32), indices

    def update_weights(self, batch_indices, batch_priorities):
        assert len(batch_indices) == len(batch_priorities)
        assert np.min(batch_priorities) > 0
        assert np.min(batch_indices) >= 0
        for idx, prio in zip(batch_indices, batch_priorities):
            idx = int(idx)
            self._it_sum[idx] = prio ** self.alpha
            self._it_min[idx] = prio ** self.alpha
        self._max_priority = max(self._max_priority, np.max(batch_priorities))

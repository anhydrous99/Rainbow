import collections
import numpy as np

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


class PrioritiedBuffer:
    def __init__(self, capacity, gamma=0.99, n_steps=2, alpha=0.6):
        self.buffer = []
        self.position = 0
        self.alpha = alpha
        self.gamma = gamma
        self.n_steps = n_steps
        self.capacity = capacity
        self.priorities = np.zeros((capacity, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position - self.n_steps]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer) - self.n_steps, batch_size, p=probabilities)

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

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return ar(states), ar(actions), ar(rewards, dtype=np.float32), ar(dones, dtype=np.uint8), ar(next_states), \
            ar(weights, dtype=np.float32), indices

    def update_weights(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

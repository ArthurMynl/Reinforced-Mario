import random
import numpy as np

def discretize(obs, bins):
    return tuple(np.digitize(x, bins) for x, bins in zip(obs, bins))

class QLearningAgent:
    def __init__(self, action_space, bins, q_table=None, alpha=0.1, gamma=0.6, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.action_space = action_space
        self.bins = bins
        self.q_table = q_table if q_table is not None else {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_state(self, observation):
        discretized_obs = discretize(observation, self.bins)
        return discretized_obs

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table.get(state, np.zeros(self.action_space.n)))

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table.get(state, np.zeros(self.action_space.n))[action]
        next_max = np.max(self.q_table.get(next_state, np.zeros(self.action_space.n)))

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        self.q_table[state][action] = new_value

        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

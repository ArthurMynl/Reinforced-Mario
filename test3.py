import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import tqdm
import numpy as np
import random
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, num_actions):
        self.model = DQN(state_size, num_actions)
        self.target_model = DQN(state_size, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set the target model to evaluation mode
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Epsilon for epsilon-greedy policy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

    def act(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                # Ensure state is a 2D tensor of shape [1, state_size]
                state = torch.FloatTensor(state).view(1, -1)  
                q_values = self.model(state)

                if q_values.shape == (1, self.num_actions):
                    return q_values.max(1)[1].item()
                else:
                    raise ValueError("Unexpected shape of q_values")
        else:
            return random.choice(range(self.num_actions))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack([torch.FloatTensor(state).view(-1) for state in states])  # Flatten each state
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack([torch.FloatTensor(state).view(-1) for state in next_states])
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.model(states).gather(1, actions).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


env = gym.make('SuperMarioBros-v3',
               apply_api_compatibility=True, render_mode=None)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

state_size = 184320  # This should be set to the actual size of your flattened state
num_actions = env.action_space.n

agent = DQNAgent(state_size, num_actions)

# Main game loop
for episode in tqdm(range(10000)):
    raw_state, _ = env.reset()
    state = raw_state.flatten()  # Flatten the image to a 1D array
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    done = False

    while not done:
        action = agent.act(state_tensor)
        raw_next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = raw_next_state.flatten()  # Flatten the next state
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        agent.store_transition(state_tensor, action, reward, next_state_tensor, done)
        state = next_state
        state_tensor = next_state_tensor  # Update the tensor for the next iteration

        agent.replay()

        if done:
            break

    if episode % 10 == 0:
        agent.update_target()

    if episode % 100 == 0:
        torch.save(agent.model.state_dict(), 'mario_dqn.pth')

env.close()

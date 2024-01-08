import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
from torch.utils.tensorboard import SummaryWriter


class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _feature_size(self, input_shape):
        return nn.Sequential(self.conv1, self.conv2, self.conv3
            ).forward(torch.zeros(1, *input_shape)).view(1, -1).size(1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Setup Environment
env = gym.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
SKIP = 4  # Define skip steps
env.frameskip = SKIP

# Initialize Network and Replay Buffer
network = DQNNetwork((1, 84, 84), env.action_space.n)
network.train()
replay_buffer = ReplayBuffer(10000)
optimizer = optim.Adam(network.parameters(), lr=0.001)
criterion = nn.MSELoss()
GAMMA = 0.99  # Discount factor


def optimize_model(batch_size):
    if len(replay_buffer) < batch_size:
        return None
    transitions = replay_buffer.sample(batch_size)
    batch = list(zip(*transitions))

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = [torch.tensor(np.array(data)) for data in batch]

    # Convert state_batch and next_state_batch to float and add a channel dimension
    state_batch = state_batch.float().unsqueeze(1)
    next_state_batch = next_state_batch.float().unsqueeze(1)

    # Forward pass through network for current and next states
    state_action_values = network(state_batch).gather(1, action_batch.unsqueeze(1).long())
    next_state_values = network(next_state_batch).max(1)[0].detach()

    # Compute expected Q values
    # Use .clone().detach() for creating new tensors, or manipulate existing tensors directly
    expected_state_action_values = (next_state_values * GAMMA) * (1 - done_batch.float()) + reward_batch.float()

    # Compute loss and optimize model
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
    



def preprocess(state):
    obs = np.dot(state[..., :3], [0.299, 0.587, 0.114])  # Convert to grayscale
    obs = np.resize(obs, (84, 84))  # Resize to 84x84
    return obs


# TensorBoard Writer
writer = SummaryWriter()

# Training Loop
total_reward = 0
losses = []
BATCH_SIZE = 128
TRAINING_STEPS = 4
total_steps = 0
done = True
games_played = 0
info = None

for step in range(15000):
    if done:
        print(f"Game {games_played} ended with reward {total_reward}")
        print(info, end='\n\n')
        state, _ = env.reset()
        games_played += 1
        writer.add_scalar('Reward/train', total_reward, total_steps)
        total_reward = 0

    # Preprocess the observation
    obs = preprocess(state)  # Assuming preprocess function converts state to grayscale and resizes it

    # Select and execute action
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        q_values = network(obs_tensor)
        action = q_values.max(1)[1].item()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(info)
    done = terminated or truncated
    total_reward += reward

    next_obs = preprocess(next_state)  # Preprocess next_state

    # Store transition in replay buffer
    replay_buffer.push(obs, action, reward, next_obs, done)

    # Perform optimization step
    if step % TRAINING_STEPS == 0:
        loss = optimize_model(BATCH_SIZE)
        if loss:
            writer.add_scalar('Loss/train', loss, total_steps)
            losses.append(loss)
        
    state = next_state
    total_steps += 1
env.close()

torch.save(network.state_dict(), 'dqn_mario_model_v1.pth')
writer.close()
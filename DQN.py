import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        self.model = DQN(state_size, num_actions)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

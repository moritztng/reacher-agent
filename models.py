import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, n_state, n_action, n_hidden = [128, 128]):
        super(Actor, self).__init__()
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.n_action= n_action
        self.bn1 = nn.BatchNorm1d(n_state)
        self.fc1 = nn.Linear(n_state, n_hidden[0])
        self.bn2 = nn.BatchNorm1d(n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.bn3 = nn.BatchNorm1d(n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], n_action)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

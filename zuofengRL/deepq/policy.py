import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, hidden, ob_space, ac_space, dueling):
        super(MLP, self).__init__()
        self.ac_space = ac_space
        self.dueling = dueling
        self.l1 = nn.Linear(ob_space, hidden)
        self.action_score = nn.Linear(hidden, ac_space)
        if self.dueling:
            self.state_value = nn.Linear(hidden, 1)

    def forward(self, x):
        hidden_unit = F.relu(self.l1(x))
        out = self.action_score(hidden_unit)
        action_score = out

        if self.dueling:
            value = self.state_value(hidden_unit)
            mean_action_score = torch.mean(action_score)
            advantage = action_score-mean_action_score
            out = value + advantage
        return out

    def choose_action(self, obs, epsilon, device):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float, device=device).view(1, -1)
            if np.random.uniform() > epsilon:
                q_value = self.forward(obs)
                a = q_value.topk(k=1, dim=1)[1].item()
            else:
                a = np.random.randint(self.ac_space)
            return a


class CNN(nn.Module):
    def __init__(self, ac_space, dueling):
        super(CNN, self).__init__()
        self.dueling = dueling
        self.ac_space = ac_space
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.l1 = nn.Linear(64*7*7, 512)
        self.l2 = nn.Linear(512, ac_space)
        if self.dueling:
            self.state_value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        hidden_unit = F.relu(self.l1(x))
        out = self.l2(hidden_unit)
        action_score = out
        if self.dueling:
            value = self.state_value(hidden_unit)
            mean_action_score = torch.mean(action_score)
            advantage = action_score - mean_action_score
            out = value + advantage
        return out

    def choose_action(self, obs, epsilon, device):
        obs = obs.__array__().transpose(2, 1, 0)
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float, device=device).view(-1, 4, 84, 84)
            if np.random.uniform() > epsilon:
                q_value = self.forward(obs)
                a = q_value.topk(k=1,dim=1)[1].item()
            else:
                a = np.random.randint(self.ac_space)
            return a


# c = CNN(4)
# print(c.choose_action(torch.randn(1,4,84,84),epsilon=0))

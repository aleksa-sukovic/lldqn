import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from models.task import Task


class TradingNetwork(nn.Module):
    def __init__(self, task: Task) -> None:
        super(TradingNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(np.prod(task.state_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(task.action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
            obs = obs.flatten(start_dim=1)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class AtariNetwork(nn.Module):
    def __init__(self, task: Task):
        super(AtariNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(task.num_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, task.action_shape)

        self.to(self.device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        obs = F.relu(self.fc1(obs.view(obs.size(0), -1)))

        return self.fc2(obs), state

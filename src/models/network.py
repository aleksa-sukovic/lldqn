import torch
import torch.nn.functional as F

from torch import nn


class Net(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.to(self.device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        obs = F.relu(self.fc1(obs.view(obs.size(0), -1)))

        return self.fc2(obs), state

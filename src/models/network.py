import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from models.task import Task


class ControlNetwork(nn.Module):
    def __init__(self, task: Task) -> None:
        super(ControlNetwork, self).__init__()
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

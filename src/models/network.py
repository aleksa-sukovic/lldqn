import numpy as np
import torch

from torch import nn

from models.task import Task


class ControlNetwork(nn.Module):
    def __init__(self, task: Task) -> None:
        super(ControlNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.init(task.code_dim, np.prod(task.action_shape))
        self.to(self.device)

    def init(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

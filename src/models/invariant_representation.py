import torch
import numpy as np

from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import TYPE_CHECKING, Any
from tianshou.data import Batch

if TYPE_CHECKING:
    from models.task import Task


class Autoencoder(nn.Module):
    def __init__(self, state_space_dim: int = 28 * 28):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, state_space_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class InvariantRepresentation:
    task: "Task"
    task_state_space: int
    autoencoder: Autoencoder

    def __init__(self, task: "Task") -> None:
        self.task = task
        self.task_state_space = self._get_state_space(task)
        self.autoencoder = None

    def train(self, batches: np.ndarray[Any, Batch]) -> None:
        dataset = self._get_dataset(batches)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        model = Autoencoder(self.task_state_space).cuda()
        num_epochs = 200
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        for epoch in range(num_epochs):
            for data in dataloader:
                batch = data[0]
                batch = torch.autograd.Variable(batch).cuda()

                output = model(batch)
                loss = criterion(output, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(
                "Epoch [{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.item())
            )

    def evaluate(self, batches: np.ndarray[Any, Batch]) -> int:
        # TODO: Ensure batches are zero-padded to match this particular task.
        pass

    def _get_dataset(self, batches: np.ndarray[Any, Batch]) -> Dataset:
        result = np.empty((batches.shape[0], self.task_state_space))
        for (index, batch) in enumerate(batches):
            experience_tuple = [
                batch.obs,
                batch.act.reshape(-1),
                batch.rew.reshape(-1),
                batch.obs_next,
            ]
            result[index] = np.concatenate(experience_tuple, axis=0)
        return TensorDataset(torch.Tensor(result))

    def _get_state_space(self, task: "Task") -> int:
        # Tuple size (current state, action, reward, successor state).
        # TODO: Properly handle differently shaped tasks.
        return task.state_shape[0] + 1 + 1 + task.state_shape[0]

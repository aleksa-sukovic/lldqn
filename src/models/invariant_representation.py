import torch
import numpy as np
import tqdm

from os.path import join
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
    def __init__(
        self, task: "Task", save_data_dir: str = "", save_model_name: str = ""
    ) -> None:
        self.task: "Task" = task
        self.task_state_space: int = self._get_state_space(task)
        self.save_data_dir = save_data_dir
        self.save_model_name = f"{save_model_name}.pt"
        self.autoencoder: Autoencoder = None
        self.train_epochs = 150
        self.train_batch = 64
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

    def train(self, batches: np.ndarray[Any, Batch]) -> None:
        dataset = self._get_dataset(batches)
        dataloader = DataLoader(dataset, batch_size=self.train_batch, shuffle=True)
        model = Autoencoder(self.task_state_space).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        progress = tqdm.tqdm(total=self.train_epochs)

        for _ in range(self.train_epochs):
            for data in dataloader:
                batch = data[0]
                batch = torch.autograd.Variable(batch).cuda()

                output = model(batch)
                loss = criterion(output, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            progress.update()
            progress.set_postfix(loss="{:.4f}".format(loss.item()))

        torch.save(model.state_dict(), join(self.save_data_dir, self.save_model_name))

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

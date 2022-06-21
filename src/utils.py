import torch
import numpy as np

from typing import TYPE_CHECKING
from torch.utils.data import Dataset, TensorDataset
from tianshou.data import to_numpy


if TYPE_CHECKING:
    from models import Task


def get_observation_dataset(task: "Task", n_episode: int = 100) -> Dataset:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = task.collector_explore.collect(n_episode=n_episode, random=True)
    batches = to_numpy(task.collector_explore.buffer)[: stats["n/st"]]
    result = torch.empty((batches.shape[0], np.prod(task.state_shape)))
    for (index, batch) in enumerate(batches):
        result[index] = torch.tensor(batch.obs)
    return TensorDataset(result.to(device))

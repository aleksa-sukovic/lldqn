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
    result = torch.empty((batches.shape[0], np.prod(batches[0].obs.shape)))
    for (index, batch) in enumerate(batches):
        result[index] = torch.tensor(batch.obs)
    return TensorDataset(result.to(device))


def get_action_dataset(task: "Task", n_sample: int = 10000) -> Dataset:
    action_count = np.prod(task.action_shape)
    sample = torch.eye(action_count)
    result = torch.empty((n_sample, action_count))
    item = 0
    for index in range(n_sample):
        result[index] = sample[item]
        item = (item + 1) % action_count
    return result

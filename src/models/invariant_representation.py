import numpy as np

from typing import TYPE_CHECKING, Any
from tianshou.data import Batch

if TYPE_CHECKING:
    from models.task import Task


class InvariantRepresentation:
    task: "Task"
    task_state_space: int

    def __init__(self, task: "Task") -> None:
        self.task = task
        self.task_state_space = self._get_state_space(task)

    def train(self, batches: np.ndarray[Any, Batch]) -> None:
        dataset = self._get_dataset(batches)

        print("Created dataset.")

    def evaluate(self, batches: np.ndarray[Any, Batch]) -> int:
        # TODO: Ensure batches are zero-padded to match this particular task.
        pass

    def _get_dataset(self, batches: np.ndarray[Any, Batch]) -> np.ndarray:
        result = np.empty((batches.shape[0], self.task_state_space))
        for (index, batch) in enumerate(batches):
            experience_tuple = [
                batch.obs,
                batch.act.reshape(-1),
                batch.rew.reshape(-1),
                batch.obs_next,
            ]
            result[index] = np.concatenate(experience_tuple, axis=0)
        np.random.shuffle(result)
        return result

    def _get_state_space(self, task: "Task") -> int:
        # Tuple size (current state, action, reward, successor state).
        # TODO: Properly handle differently shaped tasks.
        return task.state_shape[0] + 1 + 1 + task.state_shape[0]

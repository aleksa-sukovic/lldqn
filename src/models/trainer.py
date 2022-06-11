from typing import Dict, Union
from tianshou.trainer import OffpolicyTrainer

from models.task import Task

class LLDQNTrainer(OffpolicyTrainer):
    task: Task

    def __init__(self, task: Task, *args, **kwargs):
        super().__init__(
            task.policy,
            task.collector_train,
            task.collector_test,
            max_epoch=10,
            step_per_epoch=10000,
            step_per_collect=10,
            update_per_step=0.1,
            episode_per_test=100,
            batch_size=64,
            train_fn=self._train_fn,
            test_fn=self._test_fn,
            stop_fn=self._stop_fn
        )
        self.task = task

    def run(self) -> Dict[str, Union[float, str]]:
        return super().run()

    def _train_fn(self, epoch: int, env_step: int):
        self.policy.set_eps(0.1)

    def _test_fn(self, epoch: int, env_step: int):
        self.policy.set_eps(0.05)

    def _stop_fn(self, mean_rewards):
        return mean_rewards >= self.task.env.spec.reward_threshold


def lldqn_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:
    return LLDQNTrainer(*args, **kwargs).run()

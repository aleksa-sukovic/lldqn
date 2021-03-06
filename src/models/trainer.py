import wandb
import torch
import numpy as np

from os.path import join
from tianshou.trainer import OffpolicyTrainer
from tianshou.policy import BasePolicy

from models.task import Task


class DQNTrainer(OffpolicyTrainer):
    def __init__(self, task: Task, *args, **kwargs):
        super().__init__(
            task.policy,
            task.collector_train,
            task.collector_test,
            max_epoch=30,
            step_per_epoch=4096,
            step_per_collect=256,
            update_per_step=0.2,
            episode_per_test=50,
            batch_size=64,
            save_best_fn=self._save_best_fn,
            train_fn=self._train_fn,
            test_fn=self._test_fn,
            stop_fn=self._stop_fn,
            *args,
            **kwargs,
        )
        self.task = task
        self.epsilon_decay = self._decay_schedule(0.9, 0.00, 0.9, self.max_epoch)
        self.stop_fn_count = 0
        wandb.config.update({
            "train/max_epoch": 40,
            "train/step_per_epoch": 10000,
            "train/step_per_collect": 10,
            "train/update_per_step": 0.3,
            "train/episode_per_test": 100,
            "train/batch_size": 64,
            "train/decay_init": 0.9,
            "train/decay_min": 0.0,
        })

    def _save_best_fn(self, policy: BasePolicy):
        path = join(self.task.save_data_dir, self.task.save_model_name)
        torch.save(policy.state_dict(), path)

    def _train_fn(self, epoch: int, env_step: int):
        self.policy.set_eps(self.epsilon_decay[epoch])

    def _test_fn(self, epoch: int, env_step: int):
        self.policy.set_eps(0.05)

    def _stop_fn(self, mean_rewards):
        if self.task.env.spec.reward_threshold:
            if mean_rewards >= self.task.env.spec.reward_threshold:
                self.stop_fn_count = self.stop_fn_count + 1
                return self.stop_fn_count >= 3
        return False

    def _decay_schedule(
        self, init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10
    ):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)
        values = values[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        return values

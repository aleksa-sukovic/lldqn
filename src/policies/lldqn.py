import numpy as np
from typing import Union

from tianshou.data import Batch
from tianshou.policy import DQNPolicy

from policies.behavior import BehaviorPolicy


class LLDQNPolicy(DQNPolicy):
    def __init__(self, behavior_policy: BehaviorPolicy, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.behavior_policy = behavior_policy

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        return super().exploration_noise(act, batch)

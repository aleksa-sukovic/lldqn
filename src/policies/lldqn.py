import numpy as np
from typing import Union, Optional, Any

from tianshou.data import Batch
from tianshou.policy import DQNPolicy

from policies.behavior import BehaviorPolicy


class LLDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.behavior_policy: BehaviorPolicy = None
        self.last_state: Optional[Union[dict, Batch, np.ndarray]] = None

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        result = super().forward(batch, state, model, input, **kwargs)
        self.last_state = state
        return result

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            act_batch = len(act)
            rand_mask = np.random.rand(act_batch) < self.eps
            behavior_batch = self.behavior_policy(batch, self.last_state)
            act[rand_mask] = behavior_batch[rand_mask]
        return act

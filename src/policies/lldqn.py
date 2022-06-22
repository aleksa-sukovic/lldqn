import torch
import numpy as np
from typing import Union, Optional, Any, TYPE_CHECKING

from tianshou.data import Batch
from tianshou.policy import DQNPolicy

from policies.behavior import BehaviorPolicy


if TYPE_CHECKING:
    from models import Task


class LLDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task: "Task" = None
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
            with torch.no_grad():
                act_batch = len(act)
                rand_mask = np.random.rand(act_batch) < self.eps
                behavior_batch = self.behavior_policy(batch, self.last_state)
                if behavior_batch is not None:
                    logits = self.task.action_encoder.decoder(behavior_batch.act_encoded[rand_mask])
                    logits = logits.cpu().detach().numpy()
                    act[rand_mask] = logits.argmax(axis=1)
        return act

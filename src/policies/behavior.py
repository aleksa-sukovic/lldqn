import numpy as np

from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
from tianshou.policy import BasePolicy
from tianshou.data import Batch
from torch.distributions import Categorical

if TYPE_CHECKING:
    from models.task import Task


class BehaviorPolicy(BasePolicy):
    def __init__(
        self,
        tasks: List["Task"] = [],
        distribution: Categorical = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tasks = tasks
        self.distribution = distribution

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        task_index = self.distribution.sample()

        return self.tasks[task_index].policy.forward(batch, state, **kwargs)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        pass

from tianshou.trainer import OffpolicyTrainer
from typing import Dict, Union


class LLDQNTrainer(OffpolicyTrainer):
    pass


def lldqn_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:
    return LLDQNTrainer(*args, **kwargs).run()

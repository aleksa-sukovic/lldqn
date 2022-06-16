from .task import Task
from .invariant_representation import InvariantRepresentation
from .knowledge_base import KnowledgeBase
from .trainer import LLDQNTrainer
from .network import TradingNetwork, AtariNetwork
from .trainer import LLDQNTrainer

# Register custom environments.
from gym.envs.registration import register
from copy import deepcopy
from gym_anytrading.datasets import STOCKS_GOOGL

register(
    id="Stocks-v0",
    entry_point="models.envs:EvaluateStocksEnv",
    kwargs={
        'df': deepcopy(STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(STOCKS_GOOGL))
    }
)

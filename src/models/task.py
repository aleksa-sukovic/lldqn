from tianshou.data import Collector
from tianshou.env import DummyVectorEnv

from ..policies import LLDQNPolicy, BehaviorPolicy
from ..models import InvariantRepresentation, KnowledgeBase


class Task:
    env_train: DummyVectorEnv
    env_test: DummyVectorEnv
    collector_random: Collector
    collector_train: Collector
    collector_test: Collector
    policy: LLDQNPolicy
    behavior_policy: BehaviorPolicy
    invariant_representation: InvariantRepresentation
    knowledge_base: KnowledgeBase

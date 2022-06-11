import gym
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from policies import LLDQNPolicy, BehaviorPolicy
from models.invariant_representation import InvariantRepresentation
from models.knowledge_base import KnowledgeBase
from models.network import Net


class Task:
    env_train: DummyVectorEnv
    env_test: DummyVectorEnv
    collector_train: Collector
    collector_test: Collector
    policy: LLDQNPolicy
    policy_network: Net
    policy_optimizer: torch.optim.Adam
    behavior_policy: BehaviorPolicy
    invariant_representation: InvariantRepresentation
    knowledge_base: KnowledgeBase
    env: gym.Env
    state_shape: int
    action_shape: int

    def __init__(
        self,
        env_name: str,
        knowledge_base: KnowledgeBase,
        env_train_count: int = 1,
        env_test_count: int = 1,
    ) -> None:
        self.env = gym.make(env_name)
        self.env_train = DummyVectorEnv(
            [lambda: gym.make(env_name) for _ in range(env_train_count)]
        )
        self.env_test = DummyVectorEnv(
            [lambda: gym.make(env_name) for _ in range(env_test_count)]
        )
        self.state_shape = (
            self.env.observation_space.shape or self.env.observation_space.n
        )
        self.action_shape = self.env.action_space.shape or self.env.action_space.n
        self.knowledge_base = knowledge_base
        self.policy_network = Net(self.state_shape, self.action_shape)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=1e-3
        )
        self.policy = LLDQNPolicy(
            self.policy_network,
            self.policy_optimizer,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )
        self.collector_train = Collector(
            self.policy,
            self.env_train,
            VectorReplayBuffer(20000, env_train_count),
            exploration_noise=True,
        )
        self.collector_test = Collector(
            self.policy,
            self.env_test,
            exploration_noise=False,
        )

import gym
import torch

from tianshou.data import Collector, VectorReplayBuffer, to_numpy
from tianshou.env import DummyVectorEnv

from policies import LLDQNPolicy, BehaviorPolicy
from models.invariant_representation import InvariantRepresentation
from models.knowledge_base import KnowledgeBase
from models.network import Net


class Task:
    def __init__(
        self,
        env_name: str,
        knowledge_base: KnowledgeBase,
        save_data_dir: str = "",
        save_model_name: str = "TaskOne-v1",
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
        self.invariant_representation = None
        self.behavior_policy = None
        self.save_data_dir = save_data_dir
        self.save_model_name = save_model_name
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
        self.collector_explore = Collector(
            self.policy,
            self.env_test,
            VectorReplayBuffer(2000, env_test_count),
        )

    def get_invariant_representation(self) -> InvariantRepresentation:
        if not getattr(self, "invariant_representation"):
            stats = self.collector_explore.collect(n_episode=25, random=True)
            batches = to_numpy(self.collector_explore.buffer)[: stats["n/st"]]
            self.invariant_representation = InvariantRepresentation(
                task=self,
                save_data_dir=self.save_data_dir,
                save_model_name=f"{self.save_model_name}-Autoencoder",
            )
            self.invariant_representation.train(batches)
        return self.invariant_representation

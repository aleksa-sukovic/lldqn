import gym
import wandb
import torch
import numpy as np

from torch import nn
from os.path import join
from typing import Any, Dict, List, Type, Tuple
from tianshou.data import Collector, VectorReplayBuffer, to_numpy
from tianshou.env import DummyVectorEnv

from policies import LLDQNPolicy, BehaviorPolicy, BaselinePolicy
from models.autoencoder import Autoencoder
from models.knowledge_base import KnowledgeBase


class Task:
    def __init__(
        self,
        env_name: str,
        env_model: nn.Module,
        env_args: Dict[str, Any] = {},
        wrappers: List[Tuple[Type[gym.ObservationWrapper], Dict[str, Any]]] = [],
        save_data_dir: str = "",
        env_train_count: int = 1,
        env_test_count: int = 1,
        similarity_threshold: int = 0.2,
        code_dim: int = 5,
        knowledge_base: KnowledgeBase = None,
        use_baseline: bool = False,
        version: int = 1,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = env_name
        self.version = version
        self.wrappers = wrappers
        self.env_args = env_args
        self.env = self._make_env(env_name)
        self.env_train = DummyVectorEnv(
            [lambda: self._make_env(env_name) for _ in range(env_train_count)]
        )
        self.env_test = DummyVectorEnv(
            [lambda: self._make_env(env_name) for _ in range(env_test_count)]
        )
        self.action_shape = self.env.action_space.shape or self.env.action_space.n
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.save_data_dir = save_data_dir
        self.save_model_name = f"{self.name}-Policy-{self.version}.pt"
        self.knowledge_base = knowledge_base
        self.use_baseline = use_baseline
        self.similarity_threshold = similarity_threshold
        self.code_dim = code_dim
        self.policy_network = env_model(self)
        self.observation_encoder = Autoencoder(
            input_dim=np.prod(self.state_shape),
            code_dim=code_dim,
            save_data_dir=self.save_data_dir,
            save_model_name=f"{self.name}-Observation-Autoencoder.pt",
        )
        self.policy_optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=1e-3)
        if use_baseline:
            self.policy = BaselinePolicy(
                self.policy_network,
                self.policy_optimizer,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=1024,
            )
        else:
            self.policy = LLDQNPolicy(
                self.policy_network,
                self.policy_optimizer,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=1024,
            )
        self.policy.to(self.device)
        self.collector_train = Collector(
            self.policy,
            self.env_train,
            VectorReplayBuffer(100000, env_train_count),
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

    def load(self) -> None:
        self.observation_encoder.load()
        self.observation_encoder.to(self.device)

    def get_behavior_policy(self) -> BehaviorPolicy:
        logits = torch.empty(len(self.knowledge_base.tasks))

        for index, task in enumerate(self.knowledge_base.tasks):
            stats = self.collector_explore.collect(n_episode=30, random=True)
            batches = to_numpy(self.collector_explore.buffer)[: stats["n/st"]]
            loss = task.invariant_representation.evaluate(batches)
            logits[index] = loss

        mask = logits.le(self.similarity_threshold)
        probs = logits * mask
        probs[mask] = torch.nn.Softmax(dim=0)(-logits[mask])
        return BehaviorPolicy(self.knowledge_base.tasks, probs)

    def _make_env(self, env_name: str) -> gym.Env:
        env = gym.make(env_name, **self.env_args)
        for wrapper in self.wrappers:
            env = wrapper[0](env, **wrapper[1], task=self)
        return env

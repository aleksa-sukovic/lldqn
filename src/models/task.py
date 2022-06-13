import gym
import wandb
import torch

from os.path import join
from typing import Any, Dict, List, Type, Tuple
from tianshou.data import Collector, VectorReplayBuffer, to_numpy
from tianshou.env import DummyVectorEnv

from policies import LLDQNPolicy, BehaviorPolicy, BaselinePolicy
from models.invariant_representation import InvariantRepresentation
from models.knowledge_base import KnowledgeBase
from models.network import Net


class Task:
    def __init__(
        self,
        env_name: str,
        wrappers: List[Tuple[Type[gym.ObservationWrapper], Dict[str, Any]]] = [],
        save_data_dir: str = "",
        env_train_count: int = 1,
        env_test_count: int = 1,
        similarity_threshold: int = 0.2,
        knowledge_base: KnowledgeBase = None,
        use_baseline: bool = False,
        version: int = 1,
    ) -> None:
        self.name = env_name
        self.version = version
        self.wrappers = wrappers
        self.env = self._make_env(env_name)
        self.env_train = DummyVectorEnv(
            [lambda: self._make_env(env_name) for _ in range(env_train_count)]
        )
        self.env_test = DummyVectorEnv(
            [lambda: self._make_env(env_name) for _ in range(env_test_count)]
        )
        self.action_shape = self.env.action_space.shape or self.env.action_space.n
        self.save_data_dir = save_data_dir
        self.save_model_name = f"{self.name}-Policy-{self.version}.pt"
        self.knowledge_base = knowledge_base
        self.use_baseline = use_baseline
        self.similarity_threshold = similarity_threshold
        self.invariant_representation = InvariantRepresentation(
            task=self,
            save_data_dir=self.save_data_dir,
            save_model_name=f"{self.name}-Autoencoder",
        )
        self.policy_network = Net(4, self.action_shape)
        self.policy_optimizer = torch.optim.SGD(
            self.policy_network.parameters(), lr=1e-3
        )
        if use_baseline:
            self.policy = BaselinePolicy(
                self.policy_network,
                self.policy_optimizer,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
            )
        else:
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

    def load(self) -> None:
        self.invariant_representation.load()
        path = join(self.save_data_dir, self.save_model_name)
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def compile(self) -> None:
        self.invariant_representation = self.get_invariant_representation()

    def get_invariant_representation(self) -> InvariantRepresentation:
        wandb.config.update({"exploration_episodes": 30})
        stats = self.collector_explore.collect(n_episode=30, random=True)
        batches = to_numpy(self.collector_explore.buffer)[: stats["n/st"]]
        self.invariant_representation.train(batches)

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
        env = gym.make(env_name)
        for wrapper in self.wrappers:
            env = wrapper[0](env, **wrapper[1])
        return env
